#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 锁定咱们还要死磕的四个核心战场
datasets=("pubmed" "cornell" "texas" "squirrel")

# 🛡️ 保持 30 次真实期望值和公平底座
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005
TRAILS=30

echo "=========================================================="
echo "🎯 ULTIMATE K & LR SNIPER: The Final Push"
echo "=========================================================="

mkdir -p logs_ultimate
touch logs_ultimate/00_ULTIMATE_BEST.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔬 Deep diving into K and LR on: [ $ds ] <<<"
    
    case $ds in
        "pubmed")
            # 目标：突破 0.5739。既然 k=5 更好，往更小试；lr 往 0.001 附近微调
            FIXED_TAU=0.9
            FIXED_BETA=0.0001
            ks=(2 3 5)
            lrs=(0.001 0.0015 0.002)
            ;;
        "cornell")
            # 目标：突破 0.5148。k=40 更好，往下试探；lr=0.01 更好，往上微调
            FIXED_TAU=0.0
            FIXED_BETA=0.001
            ks=(30 35 40)
            lrs=(0.01 0.015 0.02)
            ;;
        "texas")
            # 目标：缩小与 0.4844 的差距。之前 k=20, lr=0.005 最好
            FIXED_TAU=0.0
            FIXED_BETA=0.0005
            ks=(20 25 30)
            lrs=(0.005 0.008 0.01)
            ;;
        "squirrel")
            # 目标：扩大 0.2133 的优势。之前 k=30, lr=0.01 最好
            FIXED_TAU=0.0
            FIXED_BETA=0.005
            ks=(30 40 50)
            lrs=(0.01 0.015 0.02)
            ;;
    esac

    best_acc=0.0
    best_config=""

    for k in "${ks[@]}"; do
        for lr in "${lrs[@]}"; do
            echo -n "   Test [k=$k, lr=$lr] (tau=$FIXED_TAU, beta=$FIXED_BETA) ... "
            
            log="logs_ultimate/Ult_${ds}_k${k}_lr${lr}.txt"
            
            python -u main.py \
                --dataset $ds \
                --method sa_ot_prompt \
                --tau $FIXED_TAU --k $k --down_lr $lr \
                --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM \
                --ot_beta $FIXED_BETA \
                --trails $TRAILS > "$log" 2>&1
                
            res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
            std=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $4}')
            
            if [ -z "$res" ]; then
                echo "❌ Failed"
            else
                echo "✅ $res ± $std"
                if (( $(echo "$res > $best_acc" | bc -l) )); then
                    best_acc=$res
                    best_config="Accuracy: $res ± $std | k=$k, lr=$lr"
                fi
            fi
        done
    done
    
    echo "🏆 Ultimate Best for $ds: $best_config (tau=$FIXED_TAU, beta=$FIXED_BETA)"
    echo "[$ds] $best_config (tau=$FIXED_TAU, beta=$FIXED_BETA)" >> logs_ultimate/00_ULTIMATE_BEST.txt
done
echo "=========================================================="
echo "🎉 ULTIMATE SNIPER COMPLETED!"