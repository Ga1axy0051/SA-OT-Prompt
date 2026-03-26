#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 8个图全面再战，Pubmed 冲刺 0.60 大关！
datasets=("pubmed" "texas" "cora" "cornell" "squirrel" "chameleon" "wisconsin" "citeseer")

EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005
TRAILS=30

echo "=========================================================="
echo "🌟 MIRACLE SQUEEZE (ROBUST): Never Give Up on OOM!"
echo "=========================================================="

mkdir -p logs_miracle
touch logs_miracle/00_MIRACLE_BEST.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔬 Miracle Grid on: [ $ds ] <<<"
    
    case $ds in
        "pubmed")
            taus=(0.95 0.99 0.999); ks=(1 2 3)
            lrs=(0.001 0.005 0.01); betas=(0.0001 0.0005) ;;
        "texas")
            taus=(0.0); ks=(24 26)
            lrs=(0.007 0.009); betas=(0.0005) ;;
        "cora")
            taus=(0.8 0.9); ks=(10 14)
            lrs=(0.0012 0.0018); betas=(0.0005) ;;
        "cornell")
            taus=(0.0); ks=(38 42)
            lrs=(0.009 0.011); betas=(0.001) ;;
        "squirrel")
            taus=(0.0); ks=(45 55)
            lrs=(0.007 0.009); betas=(0.005) ;;
        "chameleon")
            taus=(0.0); ks=(35 45)
            lrs=(0.012 0.018); betas=(0.001) ;;
        "wisconsin")
            taus=(0.0); ks=(65 75)
            lrs=(0.007 0.009); betas=(0.0005) ;;
        "citeseer")
            taus=(0.85); ks=(70 90)
            lrs=(0.0004 0.0006); betas=(0.0005) ;;
    esac

    best_acc=0.0
    best_config=""

    for t in "${taus[@]}"; do
        for k in "${ks[@]}"; do
            for lr in "${lrs[@]}"; do
                for beta in "${betas[@]}"; do
                    echo -n "   Test [tau=$t, k=$k, lr=$lr, beta=$beta] ... "
                    
                    log="logs_miracle/Mr_${ds}_t${t}_k${k}_lr${lr}_b${beta}.txt"
                    
                    # 🚀 核心死磕循环
                    while true; do
                        # 1. 检测是否已经成功跑过（断点续传）
                        if [ -f "$log" ]; then
                            res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
                            std=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $4}')
                            if [ -n "$res" ]; then
                                echo "✅ [Cached] $res ± $std"
                                if (( $(echo "$res > $best_acc" | bc -l) )); then
                                    best_acc=$res
                                    best_config="Accuracy: $res ± $std | tau=$t, k=$k, lr=$lr, ot_beta=$beta"
                                fi
                                break # 成功获取结果，跳出 while，进行下一组参数
                            fi
                        fi

                        # 2. 如果没缓存，或者缓存里没结果(说明之前OOM了)，开始执行
                        python -u main.py \
                            --dataset $ds \
                            --method sa_ot_prompt \
                            --tau $t --k $k --down_lr $lr \
                            --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM \
                            --ot_beta $beta \
                            --trails $TRAILS > "$log" 2>&1
                            
                        # 3. 检查这次跑完的结果
                        res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
                        std=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $4}')
                        
                        if [ -n "$res" ]; then
                            echo "✅ $res ± $std"
                            if (( $(echo "$res > $best_acc" | bc -l) )); then
                                best_acc=$res
                                best_config="Accuracy: $res ± $std | tau=$t, k=$k, lr=$lr, ot_beta=$beta"
                            fi
                            break # 成功跑完，跳出 while
                        else
                            # 没拿到结果，说明 OOM 暴毙了
                            echo -n "⚠️ OOM! Waiting 10s... "
                            sleep 10
                            # sleep 完后，while true 会让它自动重试这一组参数
                        fi
                    done
                done
            done
        done
    done
    
    echo "🏆 Miracle Best for $ds: $best_config"
    echo "[$ds] $best_config" >> logs_miracle/00_MIRACLE_BEST.txt
done
echo "=========================================================="
echo "🎉 MIRACLE SQUEEZE (ROBUST) COMPLETED!"