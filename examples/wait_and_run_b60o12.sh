#!/bin/bash

# 180 帧的主推理进程 PID
TARGET_PID=2967

echo "Monitor Daemon: Waiting for process $TARGET_PID to complete..."

while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 15
done

echo "Monitor Daemon: Process $TARGET_PID has terminated. Initializing b60o12 pipeline..."

# 重新授予执行权限并拉起 b60o12 推理，捕获日志
chmod +x /cloud/cloud-ssd1/cosmos-framework/examples/run_pipeline_480p_b60o12.sh
/cloud/cloud-ssd1/cosmos-framework/examples/run_pipeline_480p_b60o12.sh > /cloud/cloud-ssd1/output/run_pipeline_480p_b60o12.log 2>&1

echo "Monitor Daemon: b60o12 pipeline successfully finished."
