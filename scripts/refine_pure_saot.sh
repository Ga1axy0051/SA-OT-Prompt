#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

dataset="wisconsin"
method="sa_ot_prompt"
K=50

# 网格搜索范围
taus=(0.3 0.5 0.7)
betas=(0.01 0.001)
lrs=(0.01 0.005)

echo "=========================================================="
echo "💎 REFINING PURE SA-OT: SEARCHING FOR SOTA ON $dataset"
echo "=========================================================="

mkdir -p logs_refine

for t in "${taus[@]}"; do
    for b in "${betas[@]}"; do
        for lr in "${lrs[@]}"; do
            echo ">>> Testing: tau=$t, ot_beta=$b, lr=$lr ..."
            
            python -u main.py \
                --dataset $dataset \
                --method $method \
                --tau $t \
                --k $K \
                --ot_beta $b \
                --down_lr $lr \
                --trails 5 \
                --patience 50 > "logs_refine/Refine_${t}_${b}_${lr}.txt" 2>&1
                
            res=$(tail -n 15 "logs_refine/Refine_${t}_${b}_${lr}.txt" | grep "^Accuracy" || echo "❌ Failed")
            echo "   [Result]: $res"
        done
    done
done