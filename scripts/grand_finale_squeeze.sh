#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 8 个图整整齐齐，一家人就要在一起！
datasets=("pubmed" "texas" "cora" "cornell" "squirrel" "chameleon" "wisconsin" "citeseer")

# 🛡️ 30 次真实期望值，绝对公平的底座
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005
TRAILS=30

echo "=========================================================="
echo "🏆 GRAND FINALE SQUEEZE: The Ultimate 8-Graph Interpolation"
echo "=========================================================="

mkdir -p logs_grand
touch logs_grand/00_GRAND_BEST.txt

for ds in "${datasets[@]}"; do
    echo -e "\n>>> 🔬 Grand Interpolation on: [ $ds ] <<<"
    
    case $ds in
        # --- 之前的 5 个图 ---
        "pubmed")
            FIXED_TAU=0.9; FIXED_BETA=0.0001
            ks=(4 5 6); lrs=(0.0012 0.0015 0.0018) ;;
        "texas")
            FIXED_TAU=0.0; FIXED_BETA=0.0005
            ks=(22 25 28); lrs=(0.006 0.008 0.01) ;;
        "cora")
            FIXED_TAU=0.85; FIXED_BETA=0.0005
            ks=(12 15 18); lrs=(0.0008 0.001 0.0015) ;;
        "cornell")
            FIXED_TAU=0.0; FIXED_BETA=0.001
            ks=(40 45 50); lrs=(0.008 0.01 0.012) ;;
        "squirrel")
            FIXED_TAU=0.0; FIXED_BETA=0.005
            ks=(50 60 70); lrs=(0.008 0.01 0.012) ;;
            
        # --- 被我“吃掉”的 3 个图，现在满血吐出来 ---
        "chameleon")
            # 目标：突破 0.2388
            FIXED_TAU=0.0; FIXED_BETA=0.001
            ks=(40 50 60); lrs=(0.008 0.01 0.015) ;;
        "wisconsin")
            # 目标：扩大 0.6159 的优势
            FIXED_TAU=0.0; FIXED_BETA=0.0005
            ks=(50 60 70); lrs=(0.008 0.01 0.012) ;;
        "citeseer")
            # 目标：找回曾经 0.5377 的荣光
            FIXED_TAU=0.85; FIXED_BETA=0.0005
            ks=(80 100 120); lrs=(0.0003 0.0005 0.0008) ;;
    esac

    best_acc=0.0
    best_config=""

    for k in "${ks[@]}"; do
        for lr in "${lrs[@]}"; do
            echo -n "   Test [k=$k, lr=$lr] (tau=$FIXED_TAU, beta=$FIXED_BETA) ... "
            
            log="logs_grand/Gf_${ds}_k${k}_lr${lr}.txt"
            
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
    
    echo "🏆 Grand Best for $ds: $best_config (tau=$FIXED_TAU, beta=$FIXED_BETA)"
    echo "[$ds] $best_config (tau=$FIXED_TAU, beta=$FIXED_BETA)" >> logs_grand/00_GRAND_BEST.txt
done
echo "=========================================================="
echo "🎉 GRAND FINALE SQUEEZE COMPLETED!"