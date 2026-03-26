#!/bin/bash

# 👑 绑定空闲显卡 & 防止显存碎片化
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 咱们的两个新主场
datasets=("amazon-photo" "coauthor-cs")
methods=("fine_tune" "uniprompt" "sa_ot_prompt")

echo "=========================================================="
echo "⚔️ Reconnaissance: The New Battlegrounds"
echo "=========================================================="

mkdir -p logs_recon

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🚀 Scouting Dataset: $ds <<<"

    for method in "${methods[@]}"; do
        
        # 凭借经验盲猜的参数
        if [ "$method" == "sa_ot_prompt" ]; then
            tau=0.8  # 高度依赖原本的购买/合作网络结构
            k=20
            down_lr=0.01
        elif [ "$method" == "uniprompt" ]; then
            tau=0.5
            k=50
            down_lr=0.01
        else
            tau=0.1
            k=50
            down_lr=0.05
        fi

        echo "   -> Running $method (tau=$tau, k=$k) [5 Trails]..."

        # 跑 5 次快速摸底，patience 设为 30 加快速度
        python -u main.py \
            --dataset $ds \
            --method $method \
            --noise 0.0 \
            --tau $tau \
            --k $k \
            --down_lr $down_lr \
            --trails 5 \
            --patience 30 > "logs_recon/Recon_${ds}_${method}.txt" 2>&1
    done
done

echo -e "\n\n=========================================================="
echo "📊 RECONNAISSANCE REPORT (5 Trails Mean ± Std)"
echo "=========================================================="

for ds in "${datasets[@]}"; do
    echo "【 Dataset: $ds 】"
    for method in "${methods[@]}"; do
        res=$(tail -n 15 "logs_recon/Recon_${ds}_${method}.txt" | grep "^Accuracy" || echo "❌ Failed/Incomplete")
        printf "  %-15s : %s\n" "$method" "$res"
    done
    echo "----------------------------------------------------------"
done

echo "✅ Reconnaissance Completed!"