#!/bin/bash

# 👑 绑定显卡，确保环境纯净
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 赛道：异配图 vs 同配图
datasets=("wisconsin" "cora")
methods=("uniprompt" "sa_ot_prompt" "hybrid_prompt")

# ⚖️ 绝对公平参数：所有人共用一套标准
TAU=0.5
K=50
ALPHA=0.5
TRAILS=5
PA=(30) # 早停耐心值

echo "=========================================================="
echo "⚖️ THE ABSOLUTELY FAIR BATTLE (TAU=$TAU, K=$K)"
echo "=========================================================="

mkdir -p logs_fair

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🏟️ Stadium: $ds <<<"
    for method in "${methods[@]}"; do
        
        echo "   -> Running $method ... [Wait for it]"
        
        # 统一所有下游学习率，排除优化器干扰
        python -u main.py \
            --dataset $ds \
            --method $method \
            --tau $TAU \
            --k $K \
            --alpha $ALPHA \
            --trails $TRAILS \
            --patience 50 \
            --down_lr 0.01 > "logs_fair/Fair_${ds}_${method}.txt" 2>&1
            
        res=$(tail -n 15 "logs_fair/Fair_${ds}_${method}.txt" | grep "^Accuracy" || echo "❌ Failed")
        printf "  %-15s : %s\n" "$method" "$res"
    done
done

echo -e "\n=========================================================="
echo "✅ Fair battle completed! Time to see the real MVP."