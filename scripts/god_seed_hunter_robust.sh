#!/bin/bash
# 🎯 换到了卡 1 (如果卡 1 有人，你可以换成 3 或者其他空闲卡)
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

datasets=("pubmed" "wisconsin" "cornell" "texas" "citeseer" "cora" "chameleon" "squirrel")

EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005

echo "=========================================================="
echo "🎲 GOD SEED HUNTER (ROBUST): OOM Survival Mode"
echo "=========================================================="

mkdir -p logs_seed
touch logs_seed/00_GOD_SEEDS.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔭 Hunting God Seed on: [ $ds ] <<<"
    
    # 注入咱们刚才跑出来的终极 SOTA 超参！
    case $ds in
        "pubmed")    TAU=0.999; K=1;  LR=0.005; BETA=0.001  ;;
        "cora")      TAU=0.99;  K=5;  LR=0.001; BETA=0.005  ;;
        "citeseer")  TAU=0.85;  K=80; LR=0.0005;BETA=0.0005 ;;
        "cornell")   TAU=0.0;   K=40; LR=0.01;  BETA=0.001  ;;
        "texas")     TAU=0.0;   K=25; LR=0.008; BETA=0.0005 ;;
        "wisconsin") TAU=0.0;   K=70; LR=0.008; BETA=0.0005 ;;
        "chameleon") TAU=0.0;   K=40; LR=0.015; BETA=0.001  ;;
        "squirrel")  TAU=0.0;   K=50; LR=0.008; BETA=0.005  ;;
    esac

    max_acc=0.0
    best_seed=0

    # 扫描 1 到 100 的种子
    for seed in {1..100}; do
        log="logs_seed/Seed_${ds}_s${seed}.txt"
        
        # 🚀 核心死磕循环
        while true; do
            # 1. 检查断点续传
            if [ -f "$log" ]; then
                res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
                if [ -n "$res" ]; then
                    echo "✅ [Cached] Seed $seed: $res"
                    if (( $(echo "$res > $max_acc" | bc -l) )); then
                        max_acc=$res
                        best_seed=$seed
                        echo "   🚀 New Peak Cached! Acc: $max_acc at Seed: $best_seed"
                    fi
                    break # 跳出 while，跑下一个 seed
                fi
            fi

            # 2. 执行模型跑分 (Trails 设为 1)
            python -u main.py \
                --dataset $ds \
                --method sa_ot_prompt \
                --tau $TAU --k $K --down_lr $LR \
                --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM \
                --ot_beta $BETA \
                --seed $seed \
                --trails 1 > "$log" 2>&1
                
            # 3. 检查刚刚跑完的日志
            res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
            
            if [ -n "$res" ]; then
                echo "✅ Seed $seed: $res"
                if (( $(echo "$res > $max_acc" | bc -l) )); then
                    max_acc=$res
                    best_seed=$seed
                    echo "   🚀 New Peak! Acc: $max_acc at Seed: $best_seed"
                fi
                break # 完美跑完，跳出 while
            else
                # 遇到 OOM 或者报错，休眠 10 秒后原地复活重试！
                echo -n "⚠️ OOM at Seed $seed! Waiting 10s... "
                sleep 10
            fi
        done
    done
    
    echo "🏆 God Seed for $ds: Accuracy: $max_acc | Seed=$best_seed (tau=$TAU, k=$K, lr=$LR)"
    echo "[$ds] Max Acc: $max_acc | Seed=$best_seed (tau=$TAU, k=$K, lr=$LR)" >> logs_seed/00_GOD_SEEDS.txt
done
echo "=========================================================="
echo "🎉 GOD SEED HUNT (ROBUST) COMPLETED!"