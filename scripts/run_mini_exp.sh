#!/bin/bash

# 确保日志文件夹存在
mkdir -p logs

# 1. 定义数据集全家桶
# 同配阵营: cora, citeseer, pubmed
# 极度异配(小): texas, cornell, wisconsin
# 稠密异配(大): chameleon, squirrel
datasets=("cora" "citeseer" "pubmed" "texas" "cornell" "wisconsin" "chameleon" "squirrel")
methods=("linear_probe" "fine_tune" "sa_ot_prompt")

echo "=========================================================="
echo "🔥 SA-OT-Prompt: Full NIPS-Level Benchmark Started"
echo "=========================================================="

for dataset in "${datasets[@]}"; do
    echo -e "\n>>> 🎯 Task: $dataset"
    
    # ---------------------------------------------------------
    # 👑 参数分发引擎：参考你之前的成功设置
    # ---------------------------------------------------------
    if [[ "$dataset" =~ ^(texas|cornell|wisconsin)$ ]]; then
        # 【极稀疏异配图策略】：彻底重构结构，使用小 K 值
        tau=0.9999; k=5; ot_beta=0.1; down_lr=0.01; patience=100
    elif [[ "$dataset" =~ ^(chameleon|squirrel)$ ]]; then
        # 【中型异配图策略】：平衡原始与重构结构
        tau=0.95; k=20; ot_beta=0.05; down_lr=0.005; patience=100
    else
        # 【同配图策略 (Cora/Citeseer/Pubmed)】：保留原图，大 K 值
        tau=0.01; k=50; ot_beta=0.01; down_lr=0.001; patience=50
    fi

    for method in "${methods[@]}"; do
        echo "   [Running $method] -> tau=$tau, k=$k, lr=$down_lr"
        
        # 运行实验
        # 注意：trails 设置为 5 以保证标准差的统计意义
        python main.py \
            --dataset "$dataset" \
            --method "$method" \
            --tau "$tau" \
            --k "$k" \
            --ot_beta "$ot_beta" \
            --down_lr "$down_lr" \
            --patience "$patience" \
            --trails 5 \
            --down_epochs 500 > "logs/${dataset}_${method}.log" 2>&1
        
        # 实时从日志提取最终战报
        echo "   --------------------------------------------"
        tail -n 6 "logs/${dataset}_${method}.log" | grep -E "Accuracy|Macro F1"
        echo "   --------------------------------------------"
    done
done

echo "✅ All tasks completed. Data saved in ./logs/"