export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="/Disk2/checkpoints/rdt-finetune-1b"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/home/aloha/Github/cutlass"

export WANDB_PROJECT="rdt-1b"

# Use both GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Performance optimization settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# Run with optimized batch size and memory settings - using a simpler approach
deepspeed main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="/home/aloha/Github/RoboticsDiffusionTransformer/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=8 \
    --sample_batch_size=16 \
    --max_train_steps=200000 \
    --checkpointing_period=1000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=wandb
