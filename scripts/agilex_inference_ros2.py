#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import sys
import threading
import time
import yaml
from collections import deque

import numpy as np

# import rospy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PImage
import rclpy.time
from sensor_msgs.msg import Image, JointState, CompressedImage
from std_msgs.msg import Header
import cv2

from scripts.agilex_model import create_model
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from collections import deque

import logging
from datetime import datetime
import os

import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber

# sys.path.append("./")

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

observation_window = deque(maxlen=10)  # Keep only last 10 observations
lang_embeddings = None

# debug
preload_images = None


# Replace rospy.Rate with rclpy timer
class Rate:
    def __init__(self, frequency):
        self.period = 1.0 / frequency
        self.last_time = time.time()
    
    def sleep(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed < self.period:
            time.sleep(self.period - elapsed)
        self.last_time = time.time()


# Initialize the model
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    
    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 14,
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config


# Get the observation from the ROS topic
def get_ros_observation(args, ros_operator):
    rate = Rate(args.publish_rate)
    print_flag = True

    while True and rclpy.ok():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        # print(f"sync success when get_ros_observation")
        return (img_front, img_left, img_right,
         puppet_arm_left, puppet_arm_right)


# Update the observation window buffer
def update_observation_window(args, config, ros_operator):
    # JPEG transformation
    # Align with training
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img
    
    global observation_window
    
    print("\n=== Starting Observation Window Update ===")
    
    # Get synchronized sensor data
    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = get_ros_observation(args, ros_operator)
    
    # Debug image processing
    print(f"Processing images - Front: {img_front.shape}, Left: {img_left.shape}, Right: {img_right.shape}")
    
    # JPEG transformation
    img_front = jpeg_mapping(img_front)
    img_left = jpeg_mapping(img_left)
    img_right = jpeg_mapping(img_right)
    print("JPEG transformation completed")
    
    # Process joint positions - Take 6 values and pad with zeros to match model input
    left_pos = np.array(puppet_arm_left.position[:6])   # Take first 6 values
    right_pos = np.array(puppet_arm_right.position[:6]) # Take first 6 values
    
    # Pad each arm position with one zero to make it 7 values (matching model's expectation)
    left_pos = np.pad(left_pos, (0, 1), 'constant', constant_values=0)  # Add one zero
    right_pos = np.pad(right_pos, (0, 1), 'constant', constant_values=0)  # Add one zero
    
    # Now concatenate to get 14 values (7+7)
    qpos = np.concatenate((left_pos, right_pos), axis=0)
    qpos = torch.from_numpy(qpos).float().cuda()
    print(f"Joint positions processed (14 values): {qpos}")
    
    # Update window
    observation = {
        'qpos': qpos,
        'images': {
            config["camera_names"][0]: img_front,
            config["camera_names"][1]: img_right,
            config["camera_names"][2]: img_left,
        },
    }
    observation_window.append(observation)
    print(f"Observation window updated, current size: {len(observation_window)}")
    print("=== Observation Window Update Complete ===\n")


# RDT inference
def inference_fn(args, config, policy, t):
    global observation_window
    global lang_embeddings
    
    # print(f"Start inference_thread_fn: t={t}")
    while True and rclpy.ok():
        time1 = time.time()     

        # Debug observation window state
        print(f"Observation window size: {len(observation_window)}")

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-2]['images'][config['camera_names'][0]],
            observation_window[-2]['images'][config['camera_names'][1]],
            observation_window[-2]['images'][config['camera_names'][2]],
            
            observation_window[-1]['images'][config['camera_names'][0]],
            observation_window[-1]['images'][config['camera_names'][1]],
            observation_window[-1]['images'][config['camera_names'][2]]
        ]
        print("Images fetched from observation window")

        # fetch debug images in sequence [front, right, left]
        # image_arrs = [
        #     preload_images[config['camera_names'][0]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][2]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][1]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][0]][t],
        #     preload_images[config['camera_names'][2]][t],
        #     preload_images[config['camera_names'][1]][t]
        # ]
        # # encode the images
        # for i in range(len(image_arrs)):
        #     image_arrs[i] = cv2.imdecode(np.frombuffer(image_arrs[i], np.uint8), cv2.IMREAD_COLOR)
        # proprio = torch.from_numpy(preload_images['qpos'][t]).float().cuda()
        
        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        print("Images converted to PIL format")
        
        # for i, pos in enumerate(['f', 'r', 'l'] * 2):
        #     images[i].save(f'{t}-{i}-{pos}.png')
        
        # get last qpos in shape [14, ]
        proprio = observation_window[-1]['qpos']
        # unsqueeze to [1, 14]
        proprio = proprio.unsqueeze(0)
        print(f"Proprioception data shape: {proprio.shape}")
        
        # Model inference
        print("Running model inference...")
        # actions shaped as [1, 64, 14] in format [left, right]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings 
        ).squeeze(0).cpu().numpy()
        # print(f"inference_actions: {actions.squeeze()}")
        
        print(f"Model output actions shape: {actions.shape}")
        print(f"Action values: {actions.squeeze()}")
        print(f"Inference time: {time.time() - time1:.3f}s")
        print("=== Model Inference Complete ===\n")
        
        # print(f"Finish inference_thread_fn: t={t}")
        return actions


# Main loop for the manipulation task
def model_inference(args, config, ros_operator):
    print("\n=== Starting Robot Control Loop ===")

    global observation_window
    global lang_embeddings

    # Clear and initialize observation window
    observation_window.clear()
    
    # Load model
    print("Loading policy model...")
    # Load rdt model
    policy = make_policy(args)
    
    # Load language embeddings
    print("Loading language embeddings...")
    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"]
    
    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    # example target position: 0.07 -0.93 0.3 -0.18 -1.0 -0.1
    # example home position: 0.0 -1.822 1.554 -0.008 -1.569 -0.008
    # Initialize position of the puppet arm
    left0 = [0.0, -1.822, 1.554, -0.008, -1.569, -0.008, 0.7]
    right0 = [0.0, -1.822, 1.554, -0.008, -1.569, -0.008, 0.7]
    left1 = [0.0, -1.822, 1.554, -0.008, -1.569, -0.008, 1.5]
    right1 = [0.0, -1.822, 1.554, -0.008, -1.569, -0.008, 1.5]
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config['state_dim'])
    pre_action[:7] = np.array(left0)  # First 6 values for left arm
    pre_action[7:14] = np.array(right0)  # First 6 values for right arm
    action = None

    print("Waiting for initial observations...")
    # Get initial observations
    while len(observation_window) < 2:  # Need at least 2 observations for the model
        update_observation_window(args, config, ros_operator)
        time.sleep(0.1)

    # Inference loop
    print("Starting control loop...")
    with torch.inference_mode():
        while True and rclpy.ok():
            # The current time step
            t = 0
            rate = ros_operator.create_rate(args.publish_rate)
    
            action_buffer = np.zeros([chunk_size, config['state_dim']])
            
            while t < max_publish_step and rclpy.ok():
                # Update observation window
                update_observation_window(args, config, ros_operator)
                
                # When coming to the end of the action chunk
                if t % chunk_size == 0:
                    # Start inference
                    action_buffer = inference_fn(args, config, policy, t).copy()
                
                raw_action = action_buffer[t % chunk_size]
                action = raw_action
                # Interpolate the original action sequence
                if args.use_actions_interpolation:
                    # print(f"Time {t}, pre {pre_action}, act {action}")
                    interp_actions = interpolate_action(args, pre_action, action)
                else:
                    interp_actions = action[np.newaxis, :]

                # Execute the interpolated actions one by one
                for act in interp_actions:
                    # Take only first 6 values for each arm (excluding gripper)
                    left_action = act[:7]
                    right_action = act[7:14]
                    print(f"Executing actions - Left: {left_action}, Right: {right_action}")
                    if not args.disable_puppet_arm:
                        ros_operator.puppet_arm_publish(left_action, right_action)

                    if args.use_robot_base:
                        vel_action = act[14:16]
                        ros_operator.robot_base_publish(vel_action)
                    rate.sleep()
                    # print(f"doing action: {act}")
                t += 1
                
                print("Published Step", t)
                pre_action = action.copy()


class LogManager:
    def __init__(self, log_dir="logs"):
        # Create timestamp for unique log directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize different loggers
        self.system_logger = self._setup_logger("system", "system.log")
        self.sync_logger = self._setup_logger("sync", "sync.log")
        self.action_logger = self._setup_logger("action", "actions.log")
        self.error_logger = self._setup_logger("error", "errors.log")
        
    def _setup_logger(self, name, filename):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, filename))
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def log_sync_status(self, timestamps_dict):
        self.sync_logger.info(f"Sync Status: {timestamps_dict}")
    
    def log_action(self, action_type, data):
        self.action_logger.info(f"{action_type}: {data}")
    
    def log_error(self, error_msg, exception=None):
        if exception:
            self.error_logger.error(f"{error_msg}: {str(exception)}")
        else:
            self.error_logger.error(error_msg)
    
    def log_system(self, msg):
        self.system_logger.info(msg)

# ROS operator class
class RosOperator(Node):
    def __init__(self, args):
        super().__init__('joint_state_publisher') 
        self.args = args
        self.logger = LogManager()
        
        # Initialize basic components
        self.init()
        
        # Initialize publishers and synchronized subscribers
        self.init_publishers()
        self.init_synchronized_subscribers()
        
        # Add a timer for subscription checking
        self.create_timer(60.0, self.check_subscriptions)
        
        # Start spin thread
        self.spin_thread = threading.Thread(target=self._spin)
        self.spin_thread.daemon = True
        self.spin_thread.start()

    def _spin(self):
        """Spin ROS node in a separate thread"""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def init(self):
        """Initialize basic components"""
        self.bridge = CvBridge()
        # Single deque for synchronized data
        self.synced_data = deque(maxlen=10)
        self.robot_base_deque = deque(maxlen=2000)  # Keep this for robot base data
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def init_publishers(self):
        """Initialize all publishers"""
        # Create publishers with QoS profiles
        self.puppet_arm_left_publisher = self.create_publisher(
            JointGroupCommand,
            self.args.puppet_arm_left_cmd_topic,
            10
        )
        
        self.puppet_arm_right_publisher = self.create_publisher(
            JointGroupCommand,
            self.args.puppet_arm_right_cmd_topic,
            10
        )
        
        self.puppet_arm_left_gripper_publisher = self.create_publisher(
            JointSingleCommand,
            self.args.puppet_arm_left_gripper_cmd_topic,
            10
        )
        
        self.puppet_arm_right_gripper_publisher = self.create_publisher(
            JointSingleCommand,
            self.args.puppet_arm_right_gripper_cmd_topic,
            10
        )
        
        self.robot_base_publisher = self.create_publisher(
            Twist,
            self.args.robot_base_cmd_topic,
            10
        )

    def init_synchronized_subscribers(self):
        """Initialize synchronized subscribers using message_filters"""
        # Create QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create filtered subscribers for RGB images
        self.sub_front = message_filters.Subscriber(
            self,
            CompressedImage,
            self.args.img_front_topic,
            qos_profile=sensor_qos
        )
        self.sub_left = message_filters.Subscriber(
            self,
            CompressedImage,
            self.args.img_left_topic,
            qos_profile=sensor_qos
        )
        self.sub_right = message_filters.Subscriber(
            self,
            CompressedImage,
            self.args.img_right_topic,
            qos_profile=sensor_qos
        )
        
        # Create filtered subscribers for joint states
        self.sub_left_arm = message_filters.Subscriber(
            self,
            JointState,
            self.args.puppet_arm_left_topic,
            qos_profile=sensor_qos
        )
        self.sub_right_arm = message_filters.Subscriber(
            self,
            JointState,
            self.args.puppet_arm_right_topic,
            qos_profile=sensor_qos
        )

        # List of subscribers for RGB and arm data
        subs_rgb_arm = [
            self.sub_front,
            self.sub_left,
            self.sub_right,
            self.sub_left_arm,
            self.sub_right_arm
        ]

        # Create synchronizer for RGB and arm data
        self.ts_rgb_arm = ApproximateTimeSynchronizer(
            subs_rgb_arm,
            queue_size=10,
            slop=self.args.rgb_sync_threshold
        )
        self.ts_rgb_arm.registerCallback(self.sync_callback_rgb_arm)

        # If depth is enabled, set up depth synchronization
        if self.args.use_depth_image:
            # Create filtered subscribers for depth images
            self.sub_front_depth = message_filters.Subscriber(
                self,
                Image,
                self.args.img_front_depth_topic,
                qos_profile=sensor_qos
            )
            self.sub_left_depth = message_filters.Subscriber(
                self,
                Image,
                self.args.img_left_depth_topic,
                qos_profile=sensor_qos
            )
            self.sub_right_depth = message_filters.Subscriber(
                self,
                Image,
                self.args.img_right_depth_topic,
                qos_profile=sensor_qos
            )

            # Create synchronizer for depth data
            self.ts_depth = ApproximateTimeSynchronizer(
                [self.sub_front_depth, self.sub_left_depth, self.sub_right_depth],
                queue_size=5,
                slop=self.args.depth_sync_threshold
            )
            self.ts_depth.registerCallback(self.sync_callback_depth)

        # Subscribe to robot base data separately (not synchronized)
        if self.args.use_robot_base:
            self.create_subscription(
                Odometry,
                self.args.robot_base_topic,
                self.robot_base_callback,
                sensor_qos
            )

    def sync_callback_rgb_arm(self, msg_front, msg_left, msg_right, msg_left_arm, msg_right_arm):
        """Callback for synchronized RGB and arm data"""
        try:
            # Convert compressed images to cv2 format
            cv_front = self.compressed_imgmsg_to_cv2(msg_front)
            cv_left = self.compressed_imgmsg_to_cv2(msg_left)
            cv_right = self.compressed_imgmsg_to_cv2(msg_right)

            # Store synchronized data
            synced_data = {
                'timestamp': msg_front.header.stamp,
                'rgb_front': cv_front,
                'rgb_left': cv_left,
                'rgb_right': cv_right,
                'left_arm': msg_left_arm,
                'right_arm': msg_right_arm,
                'depth_front': None,
                'depth_left': None,
                'depth_right': None
            }
            
            self.synced_data.append(synced_data)
            # self.logger.log_sync_status("RGB-ARM sync successful")
            
        except Exception as e:
            self.logger.log_error("Error in RGB-ARM sync callback", e)

    def sync_callback_depth(self, msg_front_depth, msg_left_depth, msg_right_depth):
        """Callback for synchronized depth data"""
        try:
            # Convert depth images
            cv_front_depth = self.bridge.imgmsg_to_cv2(msg_front_depth, desired_encoding='passthrough')
            cv_left_depth = self.bridge.imgmsg_to_cv2(msg_left_depth, desired_encoding='passthrough')
            cv_right_depth = self.bridge.imgmsg_to_cv2(msg_right_depth, desired_encoding='passthrough')

            # Find matching RGB data based on timestamp
            if len(self.synced_data) > 0:
                latest_data = self.synced_data[-1]
                # Update depth data if timestamps are close enough
                time_diff = abs((msg_front_depth.header.stamp.sec - latest_data['timestamp'].sec) + 
                              (msg_front_depth.header.stamp.nanosec - latest_data['timestamp'].nanosec) * 1e-9)
                
                if time_diff < self.args.depth_sync_threshold:
                    latest_data['depth_front'] = cv_front_depth
                    latest_data['depth_left'] = cv_left_depth
                    latest_data['depth_right'] = cv_right_depth
                    # self.logger.log_sync_status("Depth sync successful")
                
        except Exception as e:
            self.logger.log_error("Error in depth sync callback", e)

    def robot_base_callback(self, msg):
        """Callback for robot base data"""
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def compressed_imgmsg_to_cv2(self, compressed_msg):
        """Convert compressed image message to CV2 format"""
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def check_subscriptions(self):
        """Periodically check if we're receiving data from subscriptions"""
        self.logger.log_system("\nSubscription Status:")
        self.logger.log_system(f"Synchronized data queue size: {len(self.synced_data)}")
        if self.args.use_robot_base:
            self.logger.log_system(f"Robot base queue size: {len(self.robot_base_deque)}")

    def get_frame(self):
        """Get latest synchronized frame data"""
        if len(self.synced_data) == 0:
            return False

        latest_data = self.synced_data[-1]
        robot_base = self.robot_base_deque[-1] if self.args.use_robot_base and len(self.robot_base_deque) > 0 else None
        
        return (
            latest_data['rgb_front'],
            latest_data['rgb_left'],
            latest_data['rgb_right'],
            latest_data['depth_front'],
            latest_data['depth_left'],
            latest_data['depth_right'],
            latest_data['left_arm'],
            latest_data['right_arm'],
            robot_base
        )

    def puppet_arm_publish(self, left, right):
        """Publish commands to puppet arms with error handling"""
        try:
            # Validate input
            if len(left) < 7 or len(right) < 7:
                raise ValueError("Invalid arm command length")
                
            # Process and publish commands
            self._publish_arm_commands(left[:6], right[:6])
            self._publish_gripper_commands(left[6], right[6])
            
        except Exception as e:
            self.logger.log_error(f"Error in puppet_arm_publish: {e}")
            raise

    def _publish_arm_commands(self, left_joints, right_joints):
        """Helper method to publish arm commands"""
        left_msg = JointGroupCommand(name="arm", cmd=left_joints)
        right_msg = JointGroupCommand(name="arm", cmd=right_joints)
        
        self.puppet_arm_left_publisher.publish(left_msg)
        self.puppet_arm_right_publisher.publish(right_msg)

    def _publish_gripper_commands(self, left_gripper, right_gripper):
        """Helper method to publish gripper commands"""
        # Clamp gripper values
        left_gripper = max(min(float(left_gripper), 1.5), 0.68)
        right_gripper = max(min(float(right_gripper), 1.5), 0.68)
        
        left_msg = JointSingleCommand(name="gripper", cmd=left_gripper)
        right_msg = JointSingleCommand(name="gripper", cmd=right_gripper)
        
        self.puppet_arm_left_gripper_publisher.publish(left_msg)
        self.puppet_arm_right_gripper_publisher.publish(right_msg)

    def robot_base_publish(self, vel):
        """Publish commands to robot base"""
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish(self, left, right):
        try:
            # Take first 6 values for joint positions and 7th value for gripper
            left_joints = left[:6]  # First 6 joint positions
            left_gripper = left[6]  # 7th value is gripper
            right_joints = right[:6]  # First 6 joint positions
            right_gripper = right[6]  # 7th value is gripper

            # protect gripper from going out of range
            left_gripper = max(min(left_gripper, 1.5), 0.68)
            right_gripper = max(min(right_gripper, 1.5), 0.68)
            
            # Debug print
            print("Publishing left arm command:", left_joints.tolist())
            print("Publishing left gripper command:", left_gripper)
            print("Publishing right arm command:", right_joints.tolist())
            print("Publishing right gripper command:", right_gripper)

            # Create and publish left arm command
            left_msg = JointGroupCommand()
            left_msg.name = "arm"  # Group name for the left arm
            left_msg.cmd = left_joints.tolist()  # Convert numpy array to list
            self.puppet_arm_left_publisher.publish(left_msg)
            
            # Create and publish left gripper command
            left_gripper_msg = JointSingleCommand()
            left_gripper_msg.name = "gripper"
            left_gripper_msg.cmd = float(left_gripper)
            self.puppet_arm_left_gripper_publisher.publish(left_gripper_msg)

            # Create and publish right arm command
            right_msg = JointGroupCommand()
            right_msg.name = "arm"  # Group name for the right arm
            right_msg.cmd = right_joints.tolist()  # Convert numpy array to list
            self.puppet_arm_right_publisher.publish(right_msg)
            
            # Create and publish right gripper command
            right_gripper_msg = JointSingleCommand()
            right_gripper_msg.name = "gripper"
            right_gripper_msg.cmd = float(right_gripper)
            self.puppet_arm_right_gripper_publisher.publish(right_gripper_msg)

            print("All arm and gripper commands published successfully")

        except Exception as e:
            print(f"Error publishing arm/gripper commands: {e}")

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        """Publish continuous commands to puppet arms until target position is reached"""
        # Take first 6 values for joints and 7th value for gripper
        left_joints = left[:6]  # Explicitly take only first 6 values for joints
        left_gripper = left[6] if len(left) > 6 else 0.7  # Get gripper value if provided
        right_joints = right[:6]  # Explicitly take only first 6 values for joints
        right_gripper = right[6] if len(right) > 6 else 0.7  # Get gripper value if provided
        
        rate = self.create_rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        left_current_gripper = None
        right_current_gripper = None
        
        # Wait for initial synchronized data
        while rclpy.ok():
            if len(self.synced_data) == 0:
                rate.sleep()
                continue
                
            latest_data = self.synced_data[-1]
            left_arm = list(latest_data['left_arm'].position[:6])  # First 6 values
            right_arm = list(latest_data['right_arm'].position[:6])  # First 6 values
            left_current_gripper = latest_data['left_arm'].position[6]
            right_current_gripper = latest_data['right_arm'].position[6]
            break

        # Calculate movement directions for joints and grippers
        left_symbol = [1 if left_joints[i] - left_arm[i] > 0 else -1 for i in range(6)]
        right_symbol = [1 if right_joints[i] - right_arm[i] > 0 else -1 for i in range(6)]
        
        left_gripper_symbol = 1 if left_gripper - left_current_gripper > 0 else -1
        right_gripper_symbol = 1 if right_gripper - right_current_gripper > 0 else -1
        
        gripper_step = 0.05

        flag = True
        step = 0
        
        while flag and rclpy.ok():
            if self.puppet_arm_publish_lock.acquire(False):
                return
                
            left_diff = [abs(left_joints[i] - left_arm[i]) for i in range(6)]
            right_diff = [abs(right_joints[i] - right_arm[i]) for i in range(6)]
            left_gripper_diff = abs(left_gripper - left_current_gripper)
            right_gripper_diff = abs(right_gripper - right_current_gripper)
            
            flag = False
            
            # Update joint positions
            for i in range(6):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left_joints[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
                    
            for i in range(6):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right_joints[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            
            # Update gripper positions smoothly
            if left_gripper_diff < gripper_step:
                left_current_gripper = left_gripper
            else:
                left_current_gripper += left_gripper_symbol * gripper_step
                flag = True
                
            if right_gripper_diff < gripper_step:
                right_current_gripper = right_gripper
            else:
                right_current_gripper += right_gripper_symbol * gripper_step
                flag = True

            # Publish arm commands (6 joints)
            left_msg = JointGroupCommand()
            left_msg.name = "arm"
            left_msg.cmd = left_arm
            self.puppet_arm_left_publisher.publish(left_msg)

            right_msg = JointGroupCommand()
            right_msg.name = "arm"
            right_msg.cmd = right_arm
            self.puppet_arm_right_publisher.publish(right_msg)
            
            # Publish gripper commands separately
            left_gripper_msg = JointSingleCommand()
            left_gripper_msg.name = "gripper"
            left_gripper_msg.cmd = float(left_current_gripper)
            self.puppet_arm_left_gripper_publisher.publish(left_gripper_msg)

            right_gripper_msg = JointSingleCommand()
            right_gripper_msg.name = "gripper"
            right_gripper_msg.cmd = float(right_current_gripper)
            self.puppet_arm_right_gripper_publisher.publish(right_gripper_msg)

            step += 1
            self.logger.log_system(f"puppet_arm_publish_continuous step {step}:")
            self.logger.log_system(f"Left arm: {left_arm}, gripper: {left_current_gripper}")
            self.logger.log_system(f"Right arm: {right_arm}, gripper: {right_current_gripper}")
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        """Start continuous publishing in a new thread with proper cleanup"""
        if hasattr(self, 'puppet_arm_publish_thread') and self.puppet_arm_publish_thread:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
        
        self.puppet_arm_publish_thread = threading.Thread(
            target=self.puppet_arm_publish_continuous, 
            args=(left, right)
        )
        self.puppet_arm_publish_thread.start()

    def robot_base_callback(self, msg):
        """Callback for robot base data"""
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    # ros 2 version:
    def init_ros(self):
        # Create subscribers
        self.create_subscription(
            CompressedImage,
            self.args.img_left_topic,
            self.img_left_callback,
            self.sensor_qos
        )
        
        self.create_subscription(
            CompressedImage,
            self.args.img_right_topic,
            self.img_right_callback,
            self.sensor_qos
        )
        
        self.create_subscription(
            CompressedImage,
            self.args.img_front_topic,
            self.img_front_callback,
            self.sensor_qos
        )

        if self.args.use_depth_image:
            self.create_subscription(
                Image,  # Changed from CompressedImage to Image
                self.args.img_left_depth_topic,
                self.img_left_depth_callback,
                self.sensor_qos
            )
            
            self.create_subscription(
                Image,  # Changed from CompressedImage to Image
                self.args.img_right_depth_topic,
                self.img_right_depth_callback,
                self.sensor_qos
            )
            
            self.create_subscription(
                Image,  # Changed from CompressedImage to Image
                self.args.img_front_depth_topic,
                self.img_front_depth_callback,
                self.sensor_qos
            )

        self.create_subscription(
            JointState,
            self.args.puppet_arm_left_topic,
            self.puppet_arm_left_callback,
            self.control_qos
        )
        
        self.create_subscription(
            JointState,
            self.args.puppet_arm_right_topic,
            self.puppet_arm_right_callback,
            self.control_qos
        )
        
        self.create_subscription(
            Odometry,
            self.args.robot_base_topic,
            self.robot_base_callback,
            self.sensor_qos
        )

        # Create publishers with debug prints
        self.puppet_arm_left_publisher = self.create_publisher(
            JointGroupCommand,
            self.args.puppet_arm_left_cmd_topic,
            10
        )
        print(f"Created left arm publisher on topic: {self.args.puppet_arm_left_cmd_topic}")
        
        self.puppet_arm_right_publisher = self.create_publisher(
            JointGroupCommand,
            self.args.puppet_arm_right_cmd_topic,
            10
        )
        print(f"Created right arm publisher on topic: {self.args.puppet_arm_right_cmd_topic}")
            
        # Add gripper publishers
        self.puppet_arm_left_gripper_publisher = self.create_publisher(
            JointSingleCommand,
            self.args.puppet_arm_left_gripper_cmd_topic,
            10
        )
        print(f"Created left gripper publisher on topic: {self.args.puppet_arm_left_gripper_cmd_topic}")
        
        self.puppet_arm_right_gripper_publisher = self.create_publisher(
            JointSingleCommand,
            self.args.puppet_arm_right_gripper_cmd_topic,
            10
        )
        print(f"Created right gripper publisher on topic: {self.args.puppet_arm_right_gripper_cmd_topic}")

        
        self.robot_base_publisher = self.create_publisher(
            Twist,
            self.args.robot_base_cmd_topic,
            10
        )

        # Verify publishers are created
        print("\nPublisher status:")
        print(f"Left arm publisher: {self.puppet_arm_left_publisher}")
        print(f"Right arm publisher: {self.puppet_arm_right_publisher}")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, 
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int, 
                        help='Random seed', default=None, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_high/camera/color/image_rect_raw/compressed', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_wrist_left/camera/color/image_rect_raw/compressed', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_wrist_right/camera/color/image_rect_raw/compressed', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_high/camera/depth/image_rect_raw', required=False)  
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_wrist_left/camera/depth/image_raw', required=False)  
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_wrist_right/camera/depth/image_raw', required=False)  
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/follower_left/commands/joint_group', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/follower_right/commands/joint_group', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/follower_left/joint_states', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/follower_right/joint_states', required=False)

    parser.add_argument('--puppet_arm_left_gripper_cmd_topic', action='store', type=str,
                    help='puppet_arm_left_gripper_cmd_topic',
                    default='/follower_left/commands/joint_single', required=False)
    parser.add_argument('--puppet_arm_right_gripper_cmd_topic', action='store', type=str,
                    help='puppet_arm_right_gripper_cmd_topic',
                    default='/follower_right/commands/joint_single', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store_true', 
                        help='Whether to use the robot base to move around',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, 
                        help='The rate at which to publish the actions',
                        default=30, required=False)
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=15, required=False)
    
    parser.add_argument('--chunk_size', action='store', type=int, 
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, 
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=True, required=False)
    parser.add_argument('--use_depth_image', action='store_true', 
                        help='Whether to use depth images',
                        default=False, required=False)
    
    parser.add_argument('--rgb_sync_threshold', type=float, default=0.1,
                    help='Sync threshold for RGB cameras and arms (seconds)')
    parser.add_argument('--depth_sync_threshold', type=float, default=0.2,
                        help='Sync threshold for depth cameras (seconds)')
    
    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging',default=False)
    
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='robotics-diffusion-transformer/rdt-1b', required=False, help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=False, 
                        default='outs/put_sponge_in_cup.pt', help='Path to the pre-encoded language instruction embeddings')
    
    args = parser.parse_args()
    return args


def main():
    rclpy.init()
    args = get_arguments()
    ros_operator = RosOperator(args)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    config = get_config(args)
    
    try:
        # Give some time for subscribers to receive initial data
        print("Waiting for initial data...")
        time.sleep(2)
        
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        print("\nShutdown requested... closing down")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        if ros_operator.spin_thread.is_alive():
            rclpy.shutdown()
            ros_operator.spin_thread.join()
        ros_operator.destroy_node()


if __name__ == '__main__':
    main()
