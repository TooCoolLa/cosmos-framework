#!/bin/bash
set -e

# 1. 激活 Conda 环境
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312

# 2. 配置 libstdc++ 补丁
export LD_PRELOAD=/usr/local/miniconda3/envs/py312/lib/libstdc++.so.6
# 按照 Gotchas 清空 LD_LIBRARY_PATH 以防 CUDA runtime 冲突
export LD_LIBRARY_PATH=''

# 3. 检查是否已经存在提取好的图片序列以跳过冗余提取
num_files=$(ls -1 /cloud/cloud-ssd1/videos/all_frames_480p 2>/dev/null | wc -l || echo 0)
if [ "$num_files" -ge 52050 ]; then
    echo "Found $num_files extracted frames in all_frames_480p. Skipping FFmpeg extraction..."
else
    echo "Cleaning old frame directory and recreating it..."
    rm -rf /cloud/cloud-ssd1/videos/all_frames_480p
    mkdir -p /cloud/cloud-ssd1/videos/all_frames_480p

    echo "Extracting 480p 30fps image frames using FFmpeg..."
    ffmpeg -i /cloud/cloud-ssd1/videos/MerachVideo10020_fixed.mp4 \
      -vf "scale=-2:480" -r 30 -q:v 2 \
      /cloud/cloud-ssd1/videos/all_frames_480p/frame_%05d.jpg
fi

# 4. 运行图片分块映射和 jsonl 准备
echo "Mapping frames to batch chunks using hardlinks..."
python /cloud/cloud-ssd1/cosmos-framework/examples/split_and_prep_480p_images.py

# 5. 运行单路多线程 Pipeline 推理 (禁用 compile 模式大幅提高小 batch 效率)
echo "Starting multi-threaded pipeline inference..."
python -m cosmos_framework.scripts.inference_multithread \
  -i /cloud/cloud-ssd1/av_batch_480p.jsonl \
  -o /cloud/cloud-ssd1/output/av_final_480p \
  --checkpoint-path /cloud/cloud-ssd1/models/cosmos3/Cosmos3-Nano \
  --no-guardrails \
  --no-use-torch-compile
