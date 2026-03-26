#!/bin/bash

# 👑 绑定空闲显卡 & 防止显存碎片化
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 选取一正一反两个典型数据集
datasets=("wisconsin" "cora")
methods=("uniprompt" "sa_ot_prompt" "hybrid_prompt")

echo "=========================================================="
echo "🧬 THE ULTIMATE TEST: HYBRID PROMPT VS THE WORLD"
echo "=========================================================="

mkdir -p logs_hybrid

for ds in "${datasets[@]}"; do
    echo -e "\n>>> ⚔️ Testing on Dataset: $ds <<<"

    for method in "${methods[@]}"; do
        
        # 1. 纯净图参数分配
        if [ "$method" == "sa_ot_prompt" ] || [ "$method" == "hybrid_prompt" ]; then
            if [ "$ds" == "cora" ]; then tau=0.9; k=50; down_lr=0.01; fi
            if [ "$ds" == "wisconsin" ]; then tau=0.3; k=50; down_lr=0.01; fi
        elif [ "$method" == "uniprompt" ]; then
            # UniPrompt 保留它最习惯的参数
            tau=0.5; k=50; down_lr=0.01
        fi

        # Hybrid 引擎专属参数
        alpha=0.5

        echo "   -> Running $method (tau=$tau, k=$k) [5 Trails]..."

        # 跑 5 次快速摸底
        python -u main.py \
            --dataset $ds \
            --method $method \
            --noise 0.0 \
            --feat_mask 0.0 \
            --tau $tau \
            --k $k \
            --down_lr $down_lr \
            --alpha $alpha \
            --trails 5 \
            --patience 30 > "logs_hybrid/Hybrid_${ds}_${method}.txt" 2>&1
            
        res=$(tail -n 15 "logs_hybrid/Hybrid_${ds}_${method}.txt" | grep "^Accuracy" || echo "❌ Failed/Incomplete")
        printf "  %-15s : %s\n" "$method" "$res"
    done
done

echo -e "\n=========================================================="
echo "✅ Hybrid test completed!"