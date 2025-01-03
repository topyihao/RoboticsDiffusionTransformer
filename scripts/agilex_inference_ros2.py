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
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
import cv2

from scripts.agilex_model import create_model
from interbotix_xs_msgs.msg import JointGroupCommand
from collections import deque

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
    left0 = [0.0, -1.822, 1.554, -0.008, -1.569, -0.008]
    right0 = [0.0, -1.822, 1.554, -0.008, -1.569, -0.008]
    left1 = [0.07, -0.93, 0.3, -0.18, -1.0, -0.1]
    right1 = [0.07, -0.93, 0.3, -0.18, -1.0, -0.1]
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config['state_dim'])
    pre_action[:6] = np.array(left0)  # First 6 values for left arm
    pre_action[7:13] = np.array(right0)  # First 6 values for right arm
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
                    left_action = act[:6]
                    right_action = act[7:13]
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


# ROS operator class
class RosOperator(Node):
    def __init__(self, args):
        super().__init__('joint_state_publisher') # update for ros 2

        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        
        # Define QoS profile for better reliability
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.control_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        print("\nInitializing ROS operator...")
        self.init()
        print("Starting ROS subscriptions and publishers...")
        self.init_ros()
        
        # Add a timer for subscription checking
        self.create_timer(60.0, self.check_subscriptions)
        
        # Create a separate thread for spinning
        self.spin_thread = threading.Thread(target=self._spin)
        self.spin_thread.daemon = True
        self.spin_thread.start()

    def _spin(self):
        """Spin ROS node in a separate thread"""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def check_subscriptions(self):
        """Periodically check if we're receiving data from subscriptions"""
        print("\nSubscription Status:")
        print(f"Left camera images: {len(self.img_left_deque)}")
        print(f"Right camera images: {len(self.img_right_deque)}")
        print(f"Front camera images: {len(self.img_front_deque)}")
        print(f"Left arm states: {len(self.puppet_arm_left_deque)}")
        print(f"Right arm states: {len(self.puppet_arm_right_deque)}")

    def puppet_arm_publish(self, left, right):
        try:
            # Take only first 6 values (excluding gripper)
            left = left[:6]  # First 6 joint positions
            right = right[:6]  # First 6 joint positions
            
            # Debug print
            print("Publishing left arm command:", left.tolist())
            print("Publishing right arm command:", right.tolist())

            # Create and publish left arm command
            left_msg = JointGroupCommand()
            left_msg.name = "arm"  # Group name for the left arm
            left_msg.cmd = left.tolist()  # Convert numpy array to list
            self.puppet_arm_left_publisher.publish(left_msg)
            print("Left arm message published successfully")

            # Create and publish right arm command
            right_msg = JointGroupCommand()
            right_msg.name = "arm"  # Group name for the right arm
            right_msg.cmd = right.tolist()  # Convert numpy array to list
            self.puppet_arm_right_publisher.publish(right_msg)
            print("Right arm message published successfully")

        except Exception as e:
            print(f"Error publishing arm commands: {e}")

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
        # Take only first 6 values
        left = left[:6]
        right = right[:6]
        
        rate = self.create_rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and rclpy.ok():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position[:6])  # First 6 values
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position[:6])  # First 6 values
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and rclpy.ok():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            
            left_msg = JointGroupCommand()
            left_msg.name = "arm"
            left_msg.cmd = left_arm
            self.puppet_arm_left_publisher.publish(left_msg)

            right_msg = JointGroupCommand()
            right_msg.name = "arm"
            right_msg.cmd = right_arm
            self.puppet_arm_right_publisher.publish(right_msg)

            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = self.create_rate(200)

        left_arm = None
        right_arm = None

        while True and rclpy.ok():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False

        # Convert timestamp to seconds for comparison
        def get_time_sec(stamp):
            return stamp.sec + stamp.nanosec * 1e-9

        if self.args.use_depth_image:
            frame_time = min([
                get_time_sec(self.img_left_deque[-1].header.stamp),
                get_time_sec(self.img_right_deque[-1].header.stamp),
                get_time_sec(self.img_front_deque[-1].header.stamp),
                get_time_sec(self.img_left_depth_deque[-1].header.stamp),
                get_time_sec(self.img_right_depth_deque[-1].header.stamp),
                get_time_sec(self.img_front_depth_deque[-1].header.stamp)
            ])
        else:
            frame_time = min([
                get_time_sec(self.img_left_deque[-1].header.stamp),
                get_time_sec(self.img_right_deque[-1].header.stamp),
                get_time_sec(self.img_front_deque[-1].header.stamp)
            ])

        # Check timestamps using the new comparison method
        if len(self.img_left_deque) == 0 or get_time_sec(self.img_left_deque[-1].header.stamp) < frame_time:
            return False
        if len(self.img_right_deque) == 0 or get_time_sec(self.img_right_deque[-1].header.stamp) < frame_time:
            return False
        if len(self.img_front_deque) == 0 or get_time_sec(self.img_front_deque[-1].header.stamp) < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or get_time_sec(self.puppet_arm_left_deque[-1].header.stamp) < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or get_time_sec(self.puppet_arm_right_deque[-1].header.stamp) < frame_time:
            return False
        
        if self.args.use_depth_image:
            if len(self.img_left_depth_deque) == 0 or get_time_sec(self.img_left_depth_deque[-1].header.stamp) < frame_time:
                return False
            if len(self.img_right_depth_deque) == 0 or get_time_sec(self.img_right_depth_deque[-1].header.stamp) < frame_time:
                return False
            if len(self.img_front_depth_deque) == 0 or get_time_sec(self.img_front_depth_deque[-1].header.stamp) < frame_time:
                return False
        
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or get_time_sec(self.robot_base_deque[-1].header.stamp) < frame_time):
            return False

        # Pop messages based on timestamp
        while get_time_sec(self.img_left_deque[0].header.stamp) < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while get_time_sec(self.img_right_deque[0].header.stamp) < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while get_time_sec(self.img_front_deque[0].header.stamp) < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while get_time_sec(self.puppet_arm_left_deque[0].header.stamp) < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while get_time_sec(self.puppet_arm_right_deque[0].header.stamp) < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while get_time_sec(self.img_left_depth_deque[0].header.stamp) < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while get_time_sec(self.img_right_depth_deque[0].header.stamp) < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while get_time_sec(self.img_front_depth_deque[0].header.stamp) < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while get_time_sec(self.robot_base_deque[0].header.stamp) < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 10:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)
        # print(f"Received left camera image, queue size: {len(self.img_left_deque)}")

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 10:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)
        # print(f"Received right camera image, queue size: {len(self.img_right_deque)}")

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 10:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)
        # print(f"Received front camera image, queue size: {len(self.img_front_deque)}")

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 10:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)
        # print(f"Received left depth image, queue size: {len(self.img_left_depth_deque)}")

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 10:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)
        # print(f"Received right depth image, queue size: {len(self.img_right_depth_deque)}")

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 10:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)
        # print(f"Received front depth image, queue size: {len(self.img_front_depth_deque)}")

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)
        # print(f"Received left puppet arm message, queue size: {len(self.puppet_arm_left_deque)}")

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)
        # print(f"Received right puppet arm message, queue size: {len(self.puppet_arm_right_deque)}")

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)
        # print(f"Received robot base message, queue size: {len(self.robot_base_deque)}")

    # ros 2 version:
    def init_ros(self):
        # Create subscribers
        self.create_subscription(
            Image,
            self.args.img_left_topic,
            self.img_left_callback,
            self.sensor_qos
        )
        
        self.create_subscription(
            Image,
            self.args.img_right_topic,
            self.img_right_callback,
            self.sensor_qos
        )
        
        self.create_subscription(
            Image,
            self.args.img_front_topic,
            self.img_front_callback,
            self.sensor_qos
        )

        if self.args.use_depth_image:
            self.create_subscription(
                Image,
                self.args.img_left_depth_topic,
                self.img_left_depth_callback,
                self.sensor_qos
            )
            
            self.create_subscription(
                Image,
                self.args.img_right_depth_topic,
                self.img_right_depth_callback,
                self.sensor_qos
            )
            
            self.create_subscription(
                Image,
                self.args.img_front_depth_topic,
                self.img_front_depth_callback,
                self.sensor_qos
            )

        self.create_subscription(
            JointState,
            self.args.puppet_arm_left_topic,
            self.puppet_arm_left_callback,
            QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE)
        )
        
        self.create_subscription(
            JointState,
            self.args.puppet_arm_right_topic,
            self.puppet_arm_right_callback,
            QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE)
        )
        
        self.create_subscription(
            Odometry,
            self.args.robot_base_topic,
            self.robot_base_callback,
            QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE)
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
                        default='/camera_high/camera/color/image_rect_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_wrist_left/camera/color/image_rect_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_wrist_right/camera/color/image_rect_raw', required=False)
    
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
                        default=25, required=False)
    
    parser.add_argument('--chunk_size', action='store', type=int, 
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, 
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true', 
                        help='Whether to use depth images',
                        default=False, required=False)
    
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
