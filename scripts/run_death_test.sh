#!/bin/bash

dataset="cora"
# 重点测试 0.8 剧毒环境，对比 0.6 作为参考
noises=(0.6 0.8) 
methods=("fine_tune" "sa_ot_prompt")

echo "=========================================================="
echo "💀 Death Test: Survival in 80% Heterophilic Noise"
echo "=========================================================="

for noise in "${noises[@]}"; do
    echo -e "\n[Noise Level: $noise]"
    
    for method in "${methods[@]}"; do
        # 针对 0.8 噪声的特殊调优
        if [ "$method" == "sa_ot_prompt" ]; then
            # 在 0.8 噪声下，我们尝试更激进地切断原图 (tau=0.9)
            tau=0.9
            k=20
            echo "   Running $method with tau=$tau (Aggressive Purge)..."
        else
            tau=0.1 # Fine-tune 不受 tau 影响，保持默认
            k=50
        fi

        # 使用 -u 实现无缓冲输出，方便实时 tail -f 查看
        python -u main.py \
            --dataset $dataset \
            --method $method \
            --noise $noise \
            --tau $tau \
            --k $k \
            --trails 5 \
            --patience 100 > "logs/death_${dataset}_${noise}_${method}.txt" 2>&1
        
        echo -n "   $method Result: "
        tail -n 15 "logs/death_${dataset}_${noise}_${method}.txt" | grep "Accuracy"
    done
done

# 自动提取纯度分析对比
echo -e "\n📊 Structural Analysis Summary (0.8 Noise):"
grep "Structural Analysis" logs/death_cora_0.8_sa_ot_prompt.txt | head -n 3