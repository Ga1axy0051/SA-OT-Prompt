#!/bin/bash

dataset="cora"
noises=(0.0 0.2 0.4 0.6) # 噪声水平：从纯净到“剧毒”
methods=("fine_tune" "sa_ot_prompt")

echo "=========================================================="
echo "🛡️ Robustness Battle: Fine-tune vs SA-OT-Prompt"
echo "=========================================================="

for noise in "${noises[@]}"; do
    echo -e "\n[Noise Level: $noise]"
    
    for method in "${methods[@]}"; do
        # 在投毒环境下，SA-OT 必须开启“排毒模式”
        if [ "$method" == "sa_ot_prompt" ]; then
            tau=0.8   # 强力过滤原图边
            k=20      # 只看最像的邻居
        else
            tau=0.1
            k=50
        fi

        python -u main.py \
            --dataset $dataset \
            --method $method \
            --noise $noise \
            --tau $tau \
            --k $k \
            --trails 5 \
            --patience 50 > "logs/robust_${dataset}_${noise}_${method}.txt" 2>&1
        
        echo -n "   $method: "
        tail -n 6 "logs/robust_${dataset}_${noise}_${method}.txt" | grep "Accuracy"
    done
done