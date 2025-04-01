#!/bin/bash

# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
DATA_TYPE="Wang_CVPR20"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="RN50_mod" # clip, RN50_mod or RN50
MASK_TYPE="spectral" # spectral, pixel, patch or nomask
BAND="all" # all, low, mid, high
RATIO=15
BATCH_SIZE=1

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
export WORLD_SIZE=1             # 设置分布式训练的总进程数为 1
export MASTER_ADDR=localhost    # 设置主节点地址为本地
export MASTER_PORT=12345        # 设置主节点端口号
export RANK=0                   # 设置当前进程的全局排名为 0
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the test command
python /root/FakeImageDetection-master/FakeImageDetection-master/test.py \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE
