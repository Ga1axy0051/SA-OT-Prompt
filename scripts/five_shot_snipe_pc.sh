#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 只狙击这两个惜败的图！
datasets=("pubmed" "cornell")

SHOTS=5
TRAILS=30
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005

echo "=========================================================="
echo "🎯 5-SHOT SNIPER: Targeting PubMed & Cornell"
echo "=========================================================="

mkdir -p logs_sniper
touch logs_sniper/00_SNIPER_BEST.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔭 Sniping Arena: [ $ds ] <<<"
    
    case $ds in
        "pubmed")
            taus=(0.8 0.85 0.9)
            ks=(1 2)
            lrs=(0.008 0.01 0.012)
            betas=(0.0005 0.001)
            ;;
        "cornell")
            taus=(0.0 0.05 0.1)
            ks=(38 40 42)
            lrs=(0.0005 0.0008 0.001)
            betas=(0.001 0.005)
            ;;
    esac

    best_acc=0.0
    best_config=""

    for t in "${taus[@]}"; do
        for k in "${ks[@]}"; do
            for lr in "${lrs[@]}"; do
                for beta in "${betas[@]}"; do
                    echo -n "   Test [tau=$t, k=$k, lr=$lr, beta=$beta] ... "
                    log="logs_sniper/Snipe_${ds}_t${t}_k${k}_lr${lr}_b${beta}.txt"
                    
                    # 🚀 防弹死磕循环
                    while true; do
                        # 断点续传
                        if [ -f "$log" ] && grep -q "^Accuracy" <(tail -n 15 "$log"); then
                            res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
                            std=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $4}')
                            echo "✅ [Cached] $res ± $std"
                            if (( $(echo "$res > $best_acc" | bc -l) )); then
                                best_acc=$res
                                best_config="Accuracy: $res ± $std | tau=$t, k=$k, lr=$lr, ot_beta=$beta"
                            fi
                            break
                        fi

                        python -u main.py \
                            --dataset $ds \
                            --method sa_ot_prompt \
                            --shot $SHOTS \
                            --tau $t --k $k --down_lr $lr \
                            --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM \
                            --ot_beta $beta \
                            --trails $TRAILS > "$log" 2>&1
                            
                        res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
                        std=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $4}')
                        
                        if [ -n "$res" ]; then
                            echo "✅ $res ± $std"
                            if (( $(echo "$res > $best_acc" | bc -l) )); then
                                best_acc=$res
                                best_config="Accuracy: $res ± $std | tau=$t, k=$k, lr=$lr, ot_beta=$beta"
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
    done
    
    echo "🏆 Sniper Best for $ds: $best_config"
    echo "[$ds] $best_config" >> logs_sniper/00_SNIPER_BEST.txt
done
echo "=========================================================="
echo "🎉 SNIPER MISSION COMPLETED!"