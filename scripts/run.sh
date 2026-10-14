#!/usr/bin/env bash
# CIFAR10 分类训练一键启动（Linux / macOS）
# 用法: bash scripts/run.sh [配置文件]
set -e

CONFIG="${1:-confits/default.yaml}"

echo "=========================================="
echo " CIFAR10 Classification Training"
echo " Config: ${CONFIG}"
echo "=========================================="

python scripts/train.py --config "${CONFIG}"
