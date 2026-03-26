#!/bin/bash

# 👑 锁定 GPU 3
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 目标：冲击 Wisconsin 的 0.5561 和 Cora 的 0.4982
datasets=("wisconsin" "cora")

# ⚖️ 绝对公平背景参数
TAU=0.5
K=50
ALPHA=0.8  # 🚀 黄金比例：80% 特征，20% 结构
TRAILS=5

echo "=========================================================="
echo "🎯 THE GOLDEN RATIO TEST (Alpha=$ALPHA, K=$K, TAU=$TAU)"
echo "=========================================================="

mkdir -p logs_golden

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔬 Refining Stadium: $ds <<<"
    
    python -u main.py \
        --dataset $ds \
        --method hybrid_prompt \
        --tau $TAU \
        --k $K \
        --alpha $ALPHA \
        --trails $TRAILS \
        --patience 50 \
        --down_lr 0.01 > "logs_golden/Golden_${ds}_alpha08.txt" 2>&1
        
    res=$(tail -n 15 "logs_golden/Golden_${ds}_alpha08.txt" | grep "^Accuracy" || echo "❌ Failed")
    echo "   [Hybrid Alpha 0.8] : $res"
done

echo -e "\n=========================================================="
echo "✅ Golden ratio test finished. Let's check the peak!"