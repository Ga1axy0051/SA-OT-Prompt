#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
datasets=("cora" "citeseer" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")
methods=("fine_tune" "uniprompt" "sa_ot_prompt")

echo "=========================================================="
echo "🏁 NIPS FINAL SHOWDOWN: 30 Trails with Golden Configs"
echo "=========================================================="

# 创建终极决战的日志文件夹
mkdir -p logs_final

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🚀 Executing Final Showdown on: $ds <<<"

    for method in "${methods[@]}"; do
        
        # 👑 为 SA-OT 注入咱们千辛万苦挖出来的天选参数！
        if [ "$method" == "sa_ot_prompt" ]; then
            if [ "$ds" == "cora" ]; then tau=0.9; k=20; down_lr=0.01; fi
            if [ "$ds" == "citeseer" ]; then tau=0.9; k=50; down_lr=0.01; fi
            if [ "$ds" == "pubmed" ]; then tau=0.5; k=50; down_lr=0.05; fi
            if [ "$ds" == "cornell" ]; then tau=0.3; k=20; down_lr=0.005; fi
            if [ "$ds" == "texas" ]; then tau=0.3; k=20; down_lr=0.01; fi
            if [ "$ds" == "wisconsin" ]; then tau=0.3; k=10; down_lr=0.01; fi
            if [ "$ds" == "chameleon" ]; then tau=0.3; k=20; down_lr=0.005; fi
            if [ "$ds" == "squirrel" ]; then tau=0.3; k=20; down_lr=0.001; fi
        
        # 为 Baseline 分配标准参数 (维持他们之前的最高水准)
        elif [ "$method" == "uniprompt" ]; then
            tau=0.5
            k=50
            down_lr=0.01 # 维持之前 UniPrompt 能跑到 0.56 的参数设定
        elif [ "$method" == "fine_tune" ]; then
            tau=0.1
            k=50
            down_lr=0.05
        fi

        echo "   -> Running $method (tau=$tau, k=$k, lr=$down_lr) [30 Trails]..."

        # 执行 30 次严谨测试
        python -u main.py \
            --dataset $ds \
            --method $method \
            --noise 0.0 \
            --tau $tau \
            --k $k \
            --down_lr $down_lr \
            --trails 30 \
            --patience 50 > "logs_final/Final_${ds}_${method}.txt" 2>&1
    done
done

echo -e "\n\n=========================================================="
echo "📊 NIPS TABLE 1: ULTIMATE RESULTS (30 Trails Mean ± Std)"
echo "=========================================================="

for ds in "${datasets[@]}"; do
    echo "【 Dataset: $ds 】"
    for method in "${methods[@]}"; do
        # 提取最后一行带有均值和方差的结果
        res=$(tail -n 15 "logs_final/Final_${ds}_${method}.txt" | grep "^Accuracy" || echo "❌ Incomplete")
        printf "  %-15s : %s\n" "$method" "$res"
    done
    echo "----------------------------------------------------------"
done

echo "✅ Final Showdown Completed! This is your Table 1."