#!/bin/bash
# 👑 锁定显卡
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

datasets=("citeseer" "cora" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")

# 🛡️ 绝对公平的底座神装 (对齐 UniPrompt 官方 Baseline)
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005
OT_BETA=0.001
TRAILS=5

echo "=========================================================="
echo "🔥 THE ULTIMATE FAIRNESS SQUEEZE: SA-OT FULL SCAN"
echo "装备: ${EPOCHS} Epochs | WD=${DOWN_WD} | Hid=${HID_DIM}"
echo "=========================================================="

mkdir -p logs_fair_squeeze
touch logs_fair_squeeze/00_BEST_RESULTS.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🚀 Deploying to Stadium: [ $ds ] <<<"
    
    # 🧠 根据图特性动态调整搜索空间，避免无效计算
    if [[ "$ds" == "cora" || "$ds" == "citeseer" || "$ds" == "pubmed" ]]; then
        # 同配图阵营：高 tau，探索不同的 k 和 lr
        taus=(0.9)
        ks=(10 20 50)
        lrs=(0.01 0.005 0.001)
    else
        # 异配图阵营：低 tau，大视野 k
        taus=(0.0 0.3 0.5)
        ks=(20 50)
        lrs=(0.05 0.01 0.005 0.001)
    fi

    best_acc=0.0
    best_config=""

    for t in "${taus[@]}"; do
        for k in "${ks[@]}"; do
            for lr in "${lrs[@]}"; do
                echo -n "   Testing [tau=$t, k=$k, lr=$lr] ... "
                
                log_file="logs_fair_squeeze/Sq_${ds}_t${t}_k${k}_lr${lr}.txt"
                
                python -u main.py \
                    --dataset $ds \
                    --method sa_ot_prompt \
                    --tau $t \
                    --k $k \
                    --down_lr $lr \
                    --down_wd $DOWN_WD \
                    --epochs $EPOCHS \
                    --hid_dim $HID_DIM \
                    --ot_beta $OT_BETA \
                    --trails $TRAILS > "$log_file" 2>&1
                    
                res=$(tail -n 15 "$log_file" | grep "^Accuracy" | awk '{print $2}')
                std=$(tail -n 15 "$log_file" | grep "^Accuracy" | awk '{print $4}')
                
                if [ -z "$res" ]; then
                    echo "❌ Failed"
                else
                    echo "✅ $res ± $std"
                    # 极其简易的最高分记录逻辑
                    if (( $(echo "$res > $best_acc" | bc -l) )); then
                        best_acc=$res
                        best_config="Accuracy: $res ± $std | tau=$t, k=$k, lr=$lr"
                    fi
                fi
            done
        done
    done
    
    echo "🏆 Best for $ds: $best_config"
    echo "[$ds] $best_config" >> logs_fair_squeeze/00_BEST_RESULTS.txt
done

echo "=========================================================="
echo "🎉 ALL DONE! Check 'logs_fair_squeeze/00_BEST_RESULTS.txt' for the holy grail."