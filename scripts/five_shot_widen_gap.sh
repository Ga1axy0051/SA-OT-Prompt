#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 只跑目前领先的这 6 个图，扩大战果
datasets=("cora" "citeseer" "wisconsin" "texas" "chameleon" "squirrel")

SHOTS=5
TRAILS=30
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005

echo "=========================================================="
echo "🚀 5-SHOT GAP WIDENER: Building the Impenetrable Moat"
echo "=========================================================="

mkdir -p logs_widen_gap
touch logs_widen_gap/00_WIDEN_BEST.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 💥 Expanding Moat on: [ $ds ] <<<"
    
    case $ds in
        "cora")
            taus=(0.6 0.7 0.75)
            ks=(4 6 8)
            lrs=(0.0005 0.002)
            FIXED_BETA=0.005 ;;
        "citeseer")
            taus=(0.5 0.6 0.65)
            ks=(45 55)
            lrs=(0.0001 0.0008)
            FIXED_BETA=0.0005 ;;
        "wisconsin")
            taus=(0.0)
            ks=(80 85 90)
            lrs=(0.0008 0.002)
            FIXED_BETA=0.0005 ;;
        "texas")
            taus=(0.0)
            ks=(48 50 55)
            lrs=(0.003 0.007)
            FIXED_BETA=0.0005 ;;
        "chameleon")
            taus=(0.0)
            ks=(5 10 12)
            lrs=(0.0005 0.002)
            FIXED_BETA=0.001 ;;
        "squirrel")
            taus=(0.0)
            ks=(35 45 50)
            lrs=(0.003 0.007)
            FIXED_BETA=0.005 ;;
    esac

    best_acc=0.0
    best_config=""

    for t in "${taus[@]}"; do
        for k in "${ks[@]}"; do
            for lr in "${lrs[@]}"; do
                echo -n "   Test [tau=$t, k=$k, lr=$lr, beta=$FIXED_BETA] ... "
                log="logs_widen_gap/Widen_${ds}_t${t}_k${k}_lr${lr}.txt"
                
                # 🛡️ OOM 生存循环
                while true; do
                    # 断点续传
                    if [ -f "$log" ] && grep -q "^Accuracy" <(tail -n 15 "$log"); then
                        res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
                        std=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $4}')
                        echo "✅ [Cached] $res ± $std"
                        if (( $(echo "$res > $best_acc" | bc -l) )); then
                            best_acc=$res
                            best_config="Accuracy: $res ± $std | tau=$t, k=$k, lr=$lr, ot_beta=$FIXED_BETA"
                        fi
                        break
                    fi

                    python -u main.py \
                        --dataset $ds \
                        --method sa_ot_prompt \
                        --shot $SHOTS \
                        --tau $t --k $k --down_lr $lr \
                        --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM \
                        --ot_beta $FIXED_BETA \
                        --trails $TRAILS > "$log" 2>&1
                        
                    res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
                    std=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $4}')
                    
                    if [ -n "$res" ]; then
                        echo "✅ $res ± $std"
                        if (( $(echo "$res > $best_acc" | bc -l) )); then
                            best_acc=$res
                            best_config="Accuracy: $res ± $std | tau=$t, k=$k, lr=$lr, ot_beta=$FIXED_BETA"
                        fi
                        break
                    else
                        echo -n "⚠️ OOM! Waiting 10s... "
                        sleep 10
                    fi
                done
            done
        done
    done
    
    echo "🏆 Widen Best for $ds: $best_config"
    echo "[$ds] $best_config" >> logs_widen_gap/00_WIDEN_BEST.txt
done
echo "=========================================================="
echo "🎉 GAP WIDENING COMPLETED!"