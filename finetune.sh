export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="/Disk2/checkpoints/rdt-finetune-1b"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/home/aloha/Github/cutlass"

export WANDB_PROJECT="robotics_diffusion_transformer"

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

# Run with optimized batch size and memory settings
python main.py \
    --pretrained_model_name_or_path="/home/aloha/Github/RoboticsDiffusionTransformer/rdt-1b" \
    --resume_from_checkpoint="/Disk2/checkpoints/rdt-finetune-1b/checkpoint-40000" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --sample_batch_size=16 \
    --max_train_steps=200000 \
    --checkpointing_period=2000 \
    --sample_period=1000 \
    --checkpoints_total_limit=20 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --use_8bit_adam \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --precomp_lang_embed \
    --report_to=wandb
