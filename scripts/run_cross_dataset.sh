#!/bin/bash

# 定义我们要征服的数据集
datasets=("citeseer" "pubmed")
noises=(0.0 0.2 0.4 0.6 0.8)
methods=("fine_tune" "sa_ot_prompt")

echo "=========================================================="
echo "🌍 Global Robustness Campaign: Cross-Dataset Death Test"
echo "=========================================================="

for ds in "${datasets[@]}"; do
    echo -e "\n\n>>> 🚀 Target Dataset: $ds <<<"
    
    for noise in "${noises[@]}"; do
        echo -e "\n[Noise Level: $noise]"
        
        for method in "${methods[@]}"; do
            # 策略调整：噪声越大，OT 越要接管
            if [ "$method" == "sa_ot_prompt" ]; then
                if [ "$noise" == "0.8" ]; then tau=0.95; else tau=0.85; fi
                k=20
            else
                tau=0.1
                k=50
            fi

            # 开始跑实验 (使用 -u 实时输出)
            python -u main.py \
                --dataset $ds \
                --method $method \
                --noise $noise \
                --tau $tau \
                --k $k \
                --trails 5 \
                --patience 100 > "logs/cross_${ds}_${noise}_${method}.txt" 2>&1
            
            echo -n "   $method Result: "
            tail -n 15 "logs/cross_${ds}_${noise}_${method}.txt" | grep "Accuracy"
        done
    done
done

echo -e "\n✅ All missions completed. Check your logs/ directory!"