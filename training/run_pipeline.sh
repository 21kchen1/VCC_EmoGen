#!/bin/bash
set -e
set -o pipefail

# 定义日志文件路径
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"


echo "=== 开始训练 ===" | tee -a $LOG_FILE
accelerate launch training/main_fe.py 2>&1 | tee -a $LOG_FILE


echo "=== 训练完成，开始推理 ===" | tee -a $LOG_FILE
accelerate launch training/accelerate_inference.py 2>&1 | tee -a $LOG_FILE


echo "=== 推理完成，开始评估 FID ===" | tee -a $LOG_FILE
python metrics/FID.py 2>&1 | tee -a $LOG_FILE

echo "=== 推理完成，开始评估 SEM ===" | tee -a $LOG_FILE
python metrics/Semantic_metrics.py 2>&1 | tee -a $LOG_FILE

echo "=== 全部任务完成 ===" | tee -a $LOG_FILE