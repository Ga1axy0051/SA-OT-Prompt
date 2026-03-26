#!/bin/bash
# 👑 锁定 GPU 3
export CUDA_VISIBLE_DEVICES=3

# 强制指定 Amazon-Photo 赛道
dataset="amazon-photo"

# 🚀 NIPS 级巅峰超参数 (已对齐归一化逻辑)
K=20              # 减少冗余边，提升单边质量
TAU=0.8           # 强力信任原始高质量同配结构
OT_BETA=0.001     # OT 辅助正则项权重
OT_EPS=0.01       # 缩小 Epsilon 让传输计划 T_star 更锋利
LR=0.01           # 稳定的下游学习率
TRAILS=5          # 5 次实验取均值，对抗 1-shot 的波动

echo "=========================================================="
echo "🥊 THE FINAL SHOWDOWN: SA-OT (Normalized) vs UniPrompt"
echo "🏟️ Stadium: $dataset | K=$K | TAU=$TAU | EPS=$OT_EPS"
echo "=========================================================="

# 1. 运行 UniPrompt (基准线)
echo ">>> Running Baseline: UniPrompt... [Wait for it]"
python -u main.py \
    --dataset $dataset \
    --method uniprompt \
    --tau $TAU \
    --k $K \
    --trails $TRAILS > logs_vs_uniprompt.txt 2>&1
uni_res=$(tail -n 15 logs_vs_uniprompt.txt | grep "^Accuracy" | awk '{print $2 " ± " $4}')

# 2. 运行 SA-OT (归一化版战神)
echo ">>> Running Challenger: SA-OT Prompt..."
python -u main.py \
    --dataset $dataset \
    --method sa_ot_prompt \
    --tau $TAU \
    --k $K \
    --ot_beta $OT_BETA \
    --ot_epsilon $OT_EPS \
    --down_lr $LR \
    --trails $TRAILS > logs_vs_saot.txt 2>&1
saot_res=$(tail -n 15 logs_vs_saot.txt | grep "^Accuracy" | awk '{print $2 " ± " $4}')

# 3. 最终战报表格输出
echo -e "\n=========================================================="
echo -e "📊 FINAL REPORT: $dataset (The Peak Performance)"
echo -e "----------------------------------------------------------"
printf "| %-18s | %-20s |\n" "Method" "Accuracy"
echo -e "|--------------------|----------------------|"
printf "| %-18s | %-20s |\n" "UniPrompt" "$uni_res"
printf "| %-18s | %-20s |\n" "SA-OT (Ours/Norm)" "$saot_res"
echo -e "=========================================================="