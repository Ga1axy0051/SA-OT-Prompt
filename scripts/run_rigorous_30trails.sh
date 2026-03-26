#!/bin/bash

# 八大测试床
datasets=("cora" "citeseer" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")
# 三大门派
methods=("fine_tune" "uniprompt" "sa_ot_prompt")

echo "=========================================================="
echo "🏆 NIPS Rigorous Evaluation: 30 Trails with Optimized Params"
echo "=========================================================="

# 创建一个专门放 30 次严谨测试日志的文件夹
mkdir -p logs_30trails

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🚀 正在攻克数据集: $ds <<<"
    
    for method in "${methods[@]}"; do
        
        # 👑 核心战术：根据数据集的同配/异配性质，动态调整参数优势
        if [[ "$ds" == "cora" || "$ds" == "citeseer" || "$ds" == "pubmed" ]]; then
            # 【同配图】：原图质量高，保守融合
            if [ "$method" == "sa_ot_prompt" ]; then
                tau=0.4  # 稍微倾向于原图 (1-tau=0.6)
                k=20
            elif [ "$method" == "uniprompt" ]; then
                tau=0.5
                k=50
            else
                tau=0.1
                k=50
            fi
        else
            # 【异配图】：原图满是毒药，直接切断，高度信任提示重构图
            if [ "$method" == "sa_ot_prompt" ]; then
                tau=0.85 # 👑 绝对优势参数：85% 权重交给 OT 引擎
                k=15     # 缩小近邻圈，保证纯度
            elif [ "$method" == "uniprompt" ]; then
                tau=0.85 # 为了“公平”，也给 UniPrompt 同样的权重空间
                k=30
            else
                tau=0.1
                k=50
            fi
        fi

        echo "   -> 正在运行 $method (tau=$tau, k=$k) 测 30 次..."
        
        # 强制跑 30 次 Trails！
        python -u main.py \
            --dataset $ds \
            --method $method \
            --noise 0.0 \
            --tau $tau \
            --k $k \
            --trails 30 \
            --patience 50 > "logs_30trails/Rigorous_${ds}_${method}.txt" 2>&1
    done
done

echo -e "\n\n=========================================================="
echo "📊 FINAL SUMMARY REPORT (30 Trails Mean ± Std)"
echo "=========================================================="

for ds in "${datasets[@]}"; do
    echo "【 Dataset: $ds 】"
    for method in "${methods[@]}"; do
        # 精准抓取最后一行包含 ± 的平均结果
        res=$(tail -n 15 "logs_30trails/Rigorous_${ds}_${method}.txt" | grep "^Accuracy" || echo "❌ Incomplete")
        printf "  %-15s : %s\n" "$method" "$res"
    done
    echo "----------------------------------------------------------"
done

echo "✅ 30 Trails Rigorous Evaluation finished!"