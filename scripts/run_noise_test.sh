#!/bin/bash

# 👑 依然绑定在空闲的 GPU 3 上，穿好防爆装甲
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

datasets=("cora" "wisconsin")
methods=("fine_tune" "uniprompt" "sa_ot_prompt")

# 设置剧毒噪声比例 60%
noise=0.6

echo "=========================================================="
echo "🌪️ THE STORM IS COMING: 60% NOISE INJECTION"
echo "=========================================================="
mkdir -p logs_noise

for ds in "${datasets[@]}"; do
    echo -e "\n>>> ☠️ Injecting $noise Noise into: $ds <<<"
    for method in "${methods[@]}"; do
        
        # 即使在恶劣环境下，也给他们用纯净图里最好的参数
        if [ "$method" == "sa_ot_prompt" ]; then
            if [ "$ds" == "cora" ]; then tau=0.9; k=20; down_lr=0.01; fi
            if [ "$ds" == "wisconsin" ]; then tau=0.3; k=10; down_lr=0.01; fi
        elif [ "$method" == "uniprompt" ]; then
            tau=0.5; k=50; down_lr=0.01
        else
            tau=0.1; k=50; down_lr=0.05
        fi

        echo "   -> Running $method at NOISE $noise [5 Trails]..."
        
        python -u main.py \
            --dataset $ds \
            --method $method \
            --noise $noise \
            --tau $tau \
            --k $k \
            --down_lr $down_lr \
            --trails 5 \
            --patience 30 > "logs_noise/Noise06_${ds}_${method}.txt" 2>&1
            
        res=$(tail -n 15 "logs_noise/Noise06_${ds}_${method}.txt" | grep "^Accuracy" || echo "❌ Failed/OOM")
        printf "  %-15s : %s\n" "$method" "$res"
    done
done

echo -e "\n=========================================================="
echo "✅ Noise robustness test completed!"