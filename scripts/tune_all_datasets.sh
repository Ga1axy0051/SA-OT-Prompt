#!/bin/bash

# 全量数据集
datasets=("cora" "citeseer" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")

echo "=========================================================="
echo "🌍 Global Grid Search: The Ultimate Parameter Optimization"
echo "=========================================================="

mkdir -p logs_tune
master_summary="logs_tune/MASTER_TOP_CONFIGS.txt"
echo "👑 NIPS Ultimate Configurations" > $master_summary

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔍 Tuning Dataset: $ds <<<"
    ds_summary="logs_tune/summary_${ds}.txt"
    > $ds_summary # 清空单数据集汇总表
    
    # 🧠 核心战术：根据数据集性质，动态切换搜索空间
    if [[ "$ds" == "cora" || "$ds" == "citeseer" || "$ds" == "pubmed" ]]; then
        # 【同配图】：原图是好人，保留大部分原图 (tau 偏大)，视野可以开阔一点 (k 偏大)
        taus=(0.5 0.7 0.9)
        ks=(20 50)
        dlrs=(0.05 0.01)
    else
        # 【异配图】：原图是毒药，斩断原图 (tau 极小)，精准近邻 (k 极小)
        taus=(0.01 0.1 0.3)
        ks=(1 10 20)
        dlrs=(0.01 0.005 0.001)
    fi

    total_runs=$(( ${#taus[@]} * ${#ks[@]} * ${#dlrs[@]} ))
    current_run=1

    for tau in "${taus[@]}"; do
        for k in "${ks[@]}"; do
            for dlr in "${dlrs[@]}"; do
                
                echo "  [$current_run/$total_runs] Testing tau=$tau | k=$k | down_lr=$dlr ..."
                log_name="logs_tune/${ds}_tau${tau}_k${k}_lr${dlr}.txt"
                
                # 为了调参效率，Trails 设为 3 快速摸底
                python -u main.py \
                    --dataset $ds \
                    --method sa_ot_prompt \
                    --noise 0.0 \
                    --tau $tau \
                    --k $k \
                    --down_lr $dlr \
                    --trails 3 \
                    --patience 30 > "$log_name" 2>&1
                
                # 抓取 Accuracy
                acc=$(tail -n 15 "$log_name" | grep "^Accuracy" || echo "Accuracy: 0.0000")
                acc_val=$(echo $acc | grep -oP 'Accuracy: \K[0-9.]+')
                
                echo -e "$acc_val\t| tau=$tau, k=$k, down_lr=$dlr \t| $acc" >> $ds_summary
                current_run=$((current_run + 1))
            done
        done
    done
    
    # 提取该数据集的第一名，写入 Master 表
    echo -e "\n🏆 Top for $ds :" >> $master_summary
    sort -nr -k1 $ds_summary | head -n 1 >> $master_summary
    echo "  >> Finished $ds. Best config saved."
done

echo -e "\n=========================================================="
echo "✅ All Datasets Tuned! Check logs_tune/MASTER_TOP_CONFIGS.txt"
echo "=========================================================="
cat $master_summary