#!/bin/bash

# 八大数据集：3大同配图 + 5大异配图
datasets=("cora" "citeseer" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")
# 三大门派：基线 vs SOTA vs 咱们
methods=("fine_tune" "uniprompt" "sa_ot_prompt")

echo "=========================================================="
echo "🏆 NIPS Table 1: The Ultimate Clean Graph Battle (8 Datasets)"
echo "=========================================================="

# 确保 logs 文件夹存在
mkdir -p logs

# 1. 跑实验阶段
for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🚀 正在攻克数据集: $ds <<<"
    
    for method in "${methods[@]}"; do
        # 针对不同方法的自适应参数
        if [ "$method" == "sa_ot_prompt" ]; then
            tau=0.5  # 纯净局，保留一半原图结构，剩下一半交给OT重构
            k=20
        elif [ "$method" == "uniprompt" ]; then
            tau=0.5  # UniPrompt 融合 KNN 图和原图
            k=50     # UniPrompt 构造 KNN 图的邻居数
        else
            tau=0.1
            k=50
        fi

        echo "   -> 正在运行 $method ..."
        # 强制 Noise=0.0，跑 5 次 Trail 取平均
        python -u main.py \
            --dataset $ds \
            --method $method \
            --noise 0.0 \
            --tau $tau \
            --k $k \
            --trails 5 \
            --patience 100 > "logs/Table1_${ds}_${method}.txt" 2>&1
    done
done

echo -e "\n\n=========================================================="
echo "📊 FINAL SUMMARY REPORT (Noise = 0.0)"
echo "=========================================================="

# 2. 战报自动汇总阶段（生成 NIPS 主表格雏形）
for ds in "${datasets[@]}"; do
    echo "【 Dataset: $ds 】"
    for method in "${methods[@]}"; do
        # 提取日志最后部分的 Accuracy (包含均值和方差)
        acc_result=$(tail -n 15 "logs/Table1_${ds}_${method}.txt" | grep "Accuracy" | head -n 1)
        
        # 如果因为报错没跑到最后，给出提示
        if [ -z "$acc_result" ]; then
            acc_result="❌ Error or Unfinished. Check logs/Table1_${ds}_${method}.txt"
        fi
        
        # 格式化输出，强迫症福音
        printf "  %-15s : %s\n" "$method" "$acc_result"
    done
    echo "----------------------------------------------------------"
done

echo "✅ All 8 Datasets for Table 1 finished!"