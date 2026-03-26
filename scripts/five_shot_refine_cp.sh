#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 专属狙击 5-shot 下的 Chameleon (k=40附近) 和 PubMed
datasets=("chameleon" "pubmed")

# 🎯 核心修正：绝对的 5-shot 领域！
SHOTS=5
TRAILS=30
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005

echo "=========================================================="
echo "🔭 5-SHOT MICROSCOPE: Deep Dive into Chameleon (k~40) & PubMed"
echo "=========================================================="

mkdir -p logs_5shot_refine
touch logs_5shot_refine/00_5SHOT_REFINE_BEST.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 💥 5-Shot Deep Search on: [ $ds ] <<<"
    
    case $ds in
        "chameleon")
            # 听你的，在 k=40 附近进行微米级扫荡！
            taus=(0.0)
            ks=(36 38 40 42 45)
            lrs=(0.005 0.01 0.015)
            betas=(0.0005 0.001 0.005)
            ;;
        "pubmed")
            # 5-shot PubMed 之前最好是 tau=0.8, k=1。咱们扩大一点 k，微调 tau
            taus=(0.8 0.85 0.9 0.95)
            ks=(1 2 3 5)
            lrs=(0.005 0.008 0.01)
            betas=(0.0005 0.001)
            ;;
    esac

    best_acc=0.0
    best_config=""

    for t in "${taus[@]}"; do
        for k in "${ks[@]}"; do
            for lr in "${lrs[@]}"; do
                for beta in "${betas[@]}"; do
                    echo -n "   Test [tau=$t, k=$k, lr=$lr, beta=$beta] ... "
                    log="logs_5shot_refine/5sRefine_${ds}_t${t}_k${k}_lr${lr}_b${beta}.txt"
                    
                    # 🛡️ 铁壁防 OOM 循环
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
    
    echo "🏆 5-Shot Refine Best for $ds: $best_config"
    echo "[$ds] $best_config" >> logs_5shot_refine/00_5SHOT_REFINE_BEST.txt
done
echo "=========================================================="
echo "🎉 5-SHOT DEEP SEARCH COMPLETED!"