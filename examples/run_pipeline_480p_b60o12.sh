#!/bin/bash
set -e

# 1. 激活 Conda 环境
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312

# 2. 配置环境补丁
export LD_PRELOAD=/usr/local/miniconda3/envs/py312/lib/libstdc++.so.6
export LD_LIBRARY_PATH=''

# 创建输出目录
mkdir -p /cloud/cloud-ssd1/output/av_final_480p_b60o12

# 3. 运行图片切块映射和 JSONL 准备
echo "Mapping frames to b60o12 batch chunks using hardlinks..."
python /cloud/cloud-ssd1/cosmos-framework/examples/split_and_prep_480p_b60o12.py

# 4. 运行单路多线程 Pipeline 推理
echo "Starting multi-threaded pipeline inference for b60o12..."
python -m cosmos_framework.scripts.inference_multithread \
  -i /cloud/cloud-ssd1/av_batch_480p_b60o12.jsonl \
  -o /cloud/cloud-ssd1/output/av_final_480p_b60o12 \
  --checkpoint-path /cloud/cloud-ssd1/models/cosmos3/Cosmos3-Nano \
  --no-guardrails \
  --no-use-torch-compile
