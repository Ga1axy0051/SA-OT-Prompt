#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

dataset="wisconsin"
# 扫描 alpha 从 0.0 (纯OT) 到 1.0 (纯UniPrompt)
alphas=(0.0 0.2 0.4 0.6 0.8 1.0)

echo "=========================================================="
echo "📈 ABLATION STUDY: Alpha Sensitivity Scan on $dataset"
echo "=========================================================="

mkdir -p logs_ablation

for a in "${alphas[@]}"; do
    echo ">>> Testing Alpha = $a ..."
    
    # 使用统一的、偏向 UniPrompt 习惯的大 K 值，确保对比公平
    python -u main.py \
        --dataset $dataset \
        --method hybrid_prompt \
        --alpha $a \
        --tau 0.2 \
        --k 50 \
        --trails 5 \
        --patience 30 > "logs_ablation/Alpha_${a}_${dataset}.txt" 2>&1
        
    res=$(tail -n 15 "logs_ablation/Alpha_${a}_${dataset}.txt" | grep "^Accuracy" || echo "❌ Failed")
    echo "   [Alpha $a] : $res"
done

echo "=========================================================="