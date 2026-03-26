#!/bin/bash

# 👑 核心换道指令：强制把当前脚本绑死在 GPU 3 上 (你可以根据 nvidia-smi 改成 1 或 2)
export CUDA_VISIBLE_DEVICES=2
# 🛡️ 碎片清理装甲：防止跑大图时 PyTorch 显存碎片化导致 OOM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

datasets=("cora" "citeseer" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")

echo "=========================================================="
echo "🌊 Deep Water Grid Search: No Stone Left Unturned"
echo "=========================================================="

mkdir -p logs_deep
master_configs="logs_deep/DEEP_GOLDEN_CONFIGS.txt"
echo "👑 NIPS Deep Squeezed Hyperparameters" > $master_configs

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🚀 Deep Scanning Dataset: $ds <<<"
    ds_log="logs_deep/summary_${ds}.txt"
    > $ds_log
    
    # 根据数据集分配深水搜索空间
    if [[ "$ds" == "cora" || "$ds" == "citeseer" || "$ds" == "pubmed" ]]; then
        taus=(0.1 0.3 0.5 0.7 0.9)
        ks=(10 20 50 100)
        dlrs=(0.05 0.01 0.005 0.001)
    else
        taus=(0.0 0.01 0.05 0.1 0.2 0.3)
        ks=(1 2 3 5 10 20)
        dlrs=(0.05 0.01 0.005 0.001)
    fi

    total_runs=$(( ${#taus[@]} * ${#ks[@]} * ${#dlrs[@]} ))
    current=1

    for tau in "${taus[@]}"; do
        for k in "${ks[@]}"; do
            for dlr in "${dlrs[@]}"; do
                
                # 打印进度和当前时间，方便你监控
                timestamp=$(date +"%Y-%m-%d %H:%M:%S")
                echo "  [$timestamp] [$current/$total_runs] Trying tau=$tau | k=$k | down_lr=$dlr ..."
                run_log="logs_deep/${ds}_t${tau}_k${k}_lr${dlr}.txt"
                
                # 保持 trails=5 用于评估参数稳定性，patience 维持 30
                python -u main.py \
                    --dataset $ds \
                    --method sa_ot_prompt \
                    --noise 0.0 \
                    --tau $tau \
                    --k $k \
                    --down_lr $dlr \
                    --trails 5 \
                    --patience 30 > "$run_log" 2>&1
                
                # 精准提取最后计算的准确率
                acc_line=$(tail -n 15 "$run_log" | grep "^Accuracy" || echo "Accuracy: 0.0000 ± 0.0000")
                acc_val=$(echo $acc_line | grep -oP 'Accuracy: \K[0-9.]+')
                
                # 将结果追加到汇总表
                echo -e "$acc_val\t| tau=$tau, k=$k, down_lr=$dlr \t| $acc_line" >> $ds_log
                current=$((current + 1))
            done
        done
    done
    
    # 扫描完一个数据集，立刻把第一名存入金榜（即使中断了也有部分结果）
    echo -e "\n🔥 Best for $ds :" >> $master_configs
    sort -nr -k1 $ds_log | head -n 1 >> $master_configs
    echo "  >> Scan complete for $ds ! Top config secured."
done

echo -e "\n=========================================================="
echo "✅ All Depths Scanned! Check logs_deep/DEEP_GOLDEN_CONFIGS.txt"
echo "=========================================================="
cat $master_configs