#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 8个图全员上阵
datasets=("cora" "citeseer" "pubmed" "chameleon" "texas" "squirrel" "wisconsin" "cornell")

SHOTS=5
TRAILS=30
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005

echo "=========================================================="
echo "🚀 5-SHOT ALL PUSH: Boundary Expansion & Low-Tau Exploration"
echo "=========================================================="

mkdir -p logs_5shot_push
touch logs_5shot_push/00_5SHOT_ALL_BEST.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔬 Pushing Limits on: [ $ds ] (5-Shot) <<<"
    
    case $ds in
        # 💡 你提议的：同质图下探 tau (Lower Tau Exploration)
        "cora")      
            taus=(0.8 0.9 0.95); betas=(0.005)
            ks=(5 10); lrs=(0.001 0.005) ;;
        "citeseer")  
            # 之前最好是0.85，咱们继续往下探！
            taus=(0.6 0.7 0.8); betas=(0.0005)
            ks=(50 60); lrs=(0.0005 0.001) ;;
        "pubmed")    
            # 之前最好是0.999，咱们试试给 OT 引擎更多权重
            taus=(0.8 0.9 0.95); betas=(0.001)
            ks=(1 3); lrs=(0.005 0.01) ;;

        # 🎯 异配图扩圈 (Boundary Push)
        "chameleon") 
            taus=(0.0); betas=(0.001)
            ks=(15 20 25); lrs=(0.001 0.003 0.005) ;;
        "texas")     
            taus=(0.0); betas=(0.0005)
            ks=(35 40 45); lrs=(0.003 0.005 0.007) ;;
        "squirrel")  
            taus=(0.0); betas=(0.005)
            ks=(20 30 40); lrs=(0.001 0.003 0.005) ;;
        "wisconsin") 
            taus=(0.0); betas=(0.0005)
            ks=(65 70 75); lrs=(0.001 0.003 0.005) ;;
        "cornell")   
            taus=(0.0); betas=(0.001)
            ks=(35 40 45); lrs=(0.001 0.003 0.005) ;;
    esac

    best_acc=0.0
    best_config=""

    for t in "${taus[@]}"; do
        for k in "${ks[@]}"; do
            for lr in "${lrs[@]}"; do
                for beta in "${betas[@]}"; do
                    echo -n "   Test [tau=$t, k=$k, lr=$lr, beta=$beta] ... "
                    log="logs_5shot_push/5all_${ds}_t${t}_k${k}_lr${lr}_b${beta}.txt"
                    
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
    
    echo "🏆 Final Push Best for $ds: $best_config"
    echo "[$ds] $best_config" >> logs_5shot_push/00_5SHOT_ALL_BEST.txt
done
echo "=========================================================="
echo "🎉 5-SHOT ALL PUSH COMPLETED!"