#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

datasets=("pubmed" "chameleon" "cora" "citeseer" "cornell" "texas" "wisconsin" "squirrel")

# 🛡️ 5-shot, 30 trails 绝对公平底座
SHOTS=5
TRAILS=30
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005

echo "=========================================================="
echo "🎯 5-SHOT SQUEEZE (30 TRAILS): The Final Sweep"
echo "=========================================================="

mkdir -p logs_5shot_search
touch logs_5shot_search/00_5SHOT_SEARCH_BEST.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔬 Grid Search on: [ $ds ] (5-Shot) <<<"
    
    case $ds in
        "pubmed")    
            taus=(0.999 0.9999)
            ks=(1 3 5)
            lrs=(0.005 0.01 0.02)
            betas=(0.001)
            ;;
        "chameleon") 
            taus=(0.0)
            ks=(30 40 50)
            lrs=(0.005 0.01 0.015)
            betas=(0.001 0.005)
            ;;
        "cora")      
            taus=(0.99 0.999)
            ks=(3 5 10)
            lrs=(0.001 0.005)
            betas=(0.005)
            ;;
        "citeseer")  
            taus=(0.85 0.9)
            ks=(60 80)
            lrs=(0.0005 0.001)
            betas=(0.0005)
            ;;
        "cornell")   
            taus=(0.0)
            ks=(30 40 50)
            lrs=(0.005 0.01)
            betas=(0.001)
            ;;
        "texas")     
            taus=(0.0)
            ks=(20 25 30)
            lrs=(0.005 0.008)
            betas=(0.0005)
            ;;
        "wisconsin") 
            taus=(0.0)
            ks=(50 70 80)
            lrs=(0.005 0.008)
            betas=(0.0005)
            ;;
        "squirrel")  
            taus=(0.0)
            ks=(40 50 60)
            lrs=(0.005 0.01)
            betas=(0.005)
            ;;
    esac

    best_acc=0.0
    best_config=""

    for t in "${taus[@]}"; do
        for k in "${ks[@]}"; do
            for lr in "${lrs[@]}"; do
                for beta in "${betas[@]}"; do
                    echo -n "   Test [tau=$t, k=$k, lr=$lr, beta=$beta] ... "
                    log="logs_5shot_search/5s_${ds}_t${t}_k${k}_lr${lr}_b${beta}.txt"
                    
                    # 🚀 核心死磕循环
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
                            
                        # 结果检查
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
    
    echo "🏆 5-Shot Best for $ds: $best_config"
    echo "[$ds] $best_config" >> logs_5shot_search/00_5SHOT_SEARCH_BEST.txt
done
echo "=========================================================="
echo "🎉 5-SHOT SEARCH COMPLETED!"