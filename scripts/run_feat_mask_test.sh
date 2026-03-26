#!/bin/bash

# 👑 稳稳绑定在 GPU 3，防爆装甲穿好
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 咱们拿异配图代表 Wisconsin 和同配图代表 Cora 开刀
datasets=("wisconsin" "cora")
methods=("fine_tune" "uniprompt" "sa_ot_prompt")

# 设定 50% 的特征丢失率
feat_mask=0.5

echo "=========================================================="
echo "🙈 FEATURE BLINDNESS TEST: 50% FEATURE MASKING"
echo "=========================================================="
mkdir -p logs_mask

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔪 Dropping 50% Features on: $ds <<<"
    for method in "${methods[@]}"; do
        
        # 依然给他们使用纯净图下的天选参数，保证公平
        if [ "$method" == "sa_ot_prompt" ]; then
            if [ "$ds" == "cora" ]; then tau=0.9; k=20; down_lr=0.01; fi
            if [ "$ds" == "wisconsin" ]; then tau=0.3; k=10; down_lr=0.01; fi
        elif [ "$method" == "uniprompt" ]; then
            tau=0.5; k=50; down_lr=0.01
        else
            tau=0.1; k=50; down_lr=0.05
        fi

        echo "   -> Running $method (feat_mask=$feat_mask) [5 Trails]..."
        
        python -u main.py \
            --dataset $ds \
            --method $method \
            --feat_mask $feat_mask \
            --tau $tau \
            --k $k \
            --down_lr $down_lr \
            --trails 5 \
            --patience 30 > "logs_mask/Mask05_${ds}_${method}.txt" 2>&1
            
        res=$(tail -n 15 "logs_mask/Mask05_${ds}_${method}.txt" | grep "^Accuracy" || echo "❌ Failed")
        printf "  %-15s : %s\n" "$method" "$res"
    done
done

echo -e "\n=========================================================="
echo "✅ Feature mask test completed!"