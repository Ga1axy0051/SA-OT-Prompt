#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 只死磕同配大图
datasets=("pubmed" "cora")

EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005
TRAILS=30

echo "=========================================================="
echo "🏔️ HOMOPHILIC LIMIT (ROBUST): Micro-Dosing with OOM Survival"
echo "=========================================================="

mkdir -p logs_homo
touch logs_homo/00_HOMO_BEST.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔬 Extreme Micro-Dosing on: [ $ds ] <<<"
    
    case $ds in
        "pubmed")
            taus=(0.999 0.9999 0.99999)
            ks=(1 2)
            lrs=(0.001 0.005)
            betas=(0.001 0.005)
            ;;
        "cora")
            taus=(0.99 0.999 0.9999)
            ks=(1 3 5)
            lrs=(0.0005 0.001)
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
                    log="logs_homo/Hm_${ds}_t${t}_k${k}_lr${lr}_b${beta}.txt"
                    
                    # 🚀 核心死磕循环
                    while true; do
                        # 1. 检查断点续传（如果之前跑出结果了，直接读缓存跳过）
                        if [ -f "$log" ]; then
                            res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
                            std=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $4}')
                            if [ -n "$res" ]; then
                                echo "✅ [Cached] $res ± $std"
                                if (( $(echo "$res > $best_acc" | bc -l) )); then
                                    best_acc=$res
                                    best_config="Accuracy: $res ± $std | tau=$t, k=$k, lr=$lr, ot_beta=$beta"
                                fi
                                break # 成功拿到结果，跳出 while 去跑下一组参数
                            fi
                        fi

                        # 2. 如果没缓存，或者缓存里没有 Accuracy（说明半路 OOM 暴毙了），开始执行
                        python -u main.py \
                            --dataset $ds \
                            --method sa_ot_prompt \
                            --tau $t --k $k --down_lr $lr \
                            --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM \
                            --ot_beta $beta \
                            --trails $TRAILS > "$log" 2>&1
                            
                        # 3. 检查刚刚跑完的日志
                        res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
                        std=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $4}')
                        
                        if [ -n "$res" ]; then
                            echo "✅ $res ± $std"
                            if (( $(echo "$res > $best_acc" | bc -l) )); then
                                best_acc=$res
                                best_config="Accuracy: $res ± $std | tau=$t, k=$k, lr=$lr, ot_beta=$beta"
                            fi
                            break # 完美跑完，跳出 while
                        else
                            # 没拿到结果，遇到 OOM 或者其他报错，原地休眠 10 秒后重试
                            echo -n "⚠️ OOM! Waiting 10s... "
                            sleep 10
                        fi
                    done
                done
            done
        done
    done
    
    echo "🏆 Homo Limit Best for $ds: $best_config"
    echo "[$ds] $best_config" >> logs_homo/00_HOMO_BEST.txt
done
echo "=========================================================="
echo "🎉 HOMOPHILIC LIMIT (ROBUST) COMPLETED!"