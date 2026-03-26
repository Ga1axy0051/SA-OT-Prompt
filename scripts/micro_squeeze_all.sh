#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 8 个数据集整整齐齐，一个不落
datasets=("cora" "citeseer" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")

echo "=========================================================="
echo "🎯 ALL-GRAPH MICRO-SQUEEZE: The Final 1% Push for NIPS"
echo "=========================================================="

mkdir -p logs_micro
touch logs_micro/00_ALL_MICRO_BEST.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔬 Micro-Surgery on: [ $ds ] <<<"
    
    # 🧠 精确制导：以昨天跑出的 SOTA 为原点，做网格微调
    case $ds in
        "cora")
            # 昨晚最佳: tau=0.9, k=20, lr=0.005
            taus=(0.9)
            ks=(15 20 25)
            lrs=(0.003 0.005 0.008)
            betas=(0.0005 0.001)
            ;;
        "citeseer")
            # 昨晚最佳: tau=0.9, k=10, lr=0.005 (注意：k>=20会OOM)
            taus=(0.85 0.9 0.95)
            ks=(80 100 120)       # 恢复大视野，这才是 OT 引擎在 Citeseer 上的真实实力
            lrs=(0.0005 0.001 0.003)
            betas=(0.0005 0.001)
            ;;
        "pubmed")
            # 昨晚最佳: tau=0.9, k=10, lr=0.001
            taus=(0.9)
            ks=(8 10 15)
            lrs=(0.0005 0.001 0.003)
            betas=(0.0005 0.001)
            ;;
        "cornell")
            # 昨晚最佳: tau=0.0, k=50, lr=0.01
            taus=(0.0 0.05)
            ks=(40 50 60)
            lrs=(0.008 0.01 0.02)
            betas=(0.0005 0.001)
            ;;
        "texas")
            # 昨晚最佳: tau=0.0, k=20, lr=0.005
            taus=(0.0 0.05)
            ks=(15 20 25)
            lrs=(0.003 0.005 0.008)
            betas=(0.0005 0.001)
            ;;
        "wisconsin")
            # 昨晚最佳: tau=0.0, k=50, lr=0.01
            taus=(0.0 0.05)
            ks=(40 50 60)
            lrs=(0.008 0.01 0.02)
            betas=(0.0005 0.001)
            ;;
        "chameleon")
            # 昨晚最佳: tau=0.0, k=50, lr=0.01
            taus=(0.0 0.05)
            ks=(40 50 60)
            lrs=(0.008 0.01 0.02)
            betas=(0.0005 0.001)
            ;;
        "squirrel")
            # 昨晚最佳: tau=0.0, k=20, lr=0.05
            taus=(0.0 0.05)
            ks=(15 20 25)
            lrs=(0.03 0.05 0.08)
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
                    
                    log="logs_micro/Sq_${ds}_t${t}_k${k}_lr${lr}_b${beta}.txt"
                    
                    python -u main.py \
                        --dataset $ds \
                        --method sa_ot_prompt \
                        --tau $t --k $k --down_lr $lr \
                        --down_wd 0.00005 --epochs 2000 --hid_dim 256 \
                        --ot_beta $beta \
                        --trails 5 > "$log" 2>&1
                        
                    res=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $2}')
                    std=$(tail -n 15 "$log" | grep "^Accuracy" | awk '{print $4}')
                    
                    if [ -z "$res" ]; then
                        echo "❌ Failed (OOM?)"
                    else
                        echo "✅ $res ± $std"
                        if (( $(echo "$res > $best_acc" | bc -l) )); then
                            best_acc=$res
                            best_config="Accuracy: $res ± $std | tau=$t, k=$k, lr=$lr, ot_beta=$beta"
                        fi
                    fi
                done
            done
        done
    done
    
    echo "🏆 Micro Best for $ds: $best_config"
    echo "[$ds] $best_config" >> logs_micro/00_ALL_MICRO_BEST.txt
done
echo "=========================================================="
echo "🎉 ALL-GRAPH MICRO-SQUEEZE COMPLETED!"