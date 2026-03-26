#!/bin/bash
# 👑 锁定显卡
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

datasets=("cora" "citeseer" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")
methods=("fine_tune" "uniprompt" "sa_ot_prompt")

echo "=========================================================="
echo "🏆 NIPS MAIN TABLE: SA-OT (Ours) vs UniPrompt (Official)"
echo "=========================================================="

mkdir -p logs_main

for ds in "${datasets[@]}"; do
    echo -e "\n=========================================================="
    echo "🏟️  Stadium: $ds "
    
    # ---------------------------------------------------------
    # 🧠 1. 注入 SA-OT (Ours) 的巅峰参数
    # ---------------------------------------------------------
    case $ds in
        "cora")       SA_TAU=0.9; SA_K=20;  SA_LR=0.005; SA_BETA=0.01 ;;
        "citeseer")   SA_TAU=0.9; SA_K=100; SA_LR=0.001; SA_BETA=0.01 ;;
        "pubmed")     SA_TAU=0.9; SA_K=10;  SA_LR=0.001;  SA_BETA=0.01 ;;
        
        # 👇 异配图强推 k=50 和 ot_beta=0.001 组合
        "cornell")    SA_TAU=0.5; SA_K=50;  SA_LR=0.01;  SA_BETA=0.001 ;; 
        "wisconsin")  SA_TAU=0.5; SA_K=50;  SA_LR=0.01;  SA_BETA=0.001 ;; 
        "texas")      SA_TAU=0.3; SA_K=20;  SA_LR=0.001; SA_BETA=0.01 ;; # Texas 刚才 0.449 已经赢麻了，保持原样！
        
        "chameleon")  SA_TAU=0.0; SA_K=50;  SA_LR=0.01;  SA_BETA=0.001 ;;
        "squirrel")   SA_TAU=0.3; SA_K=50;  SA_LR=0.005; SA_BETA=0.001 ;;
    esac

    # ---------------------------------------------------------
    # 🛡️ 2. 注入 UniPrompt (Official GraphMAE) 的官方参数
    # ---------------------------------------------------------
    case $ds in
        "cora")       UNI_TAU=0.9999;  UNI_K=50; UNI_LR=0.0005 ;;
        "citeseer")   UNI_TAU=0.99999; UNI_K=1;  UNI_LR=0.0001 ;;
        "pubmed")     UNI_TAU=0.999;   UNI_K=1;  UNI_LR=0.01   ;;
        "cornell")    UNI_TAU=0.9999;  UNI_K=50; UNI_LR=0.05   ;;
        "texas")      UNI_TAU=0.9999;  UNI_K=50; UNI_LR=0.0005 ;;
        "wisconsin")  UNI_TAU=0.9999;  UNI_K=50; UNI_LR=0.01   ;;
        "chameleon")  UNI_TAU=0.99999; UNI_K=50; UNI_LR=0.005  ;;
        "squirrel")   UNI_TAU=0.9999;  UNI_K=50; UNI_LR=0.05   ;;
    esac

    for method in "${methods[@]}"; do
        echo -n "   -> Running $method ... "
        
        if [ "$method" == "sa_ot_prompt" ]; then
            python -u main.py --dataset $ds --method $method --tau $SA_TAU --k $SA_K --down_lr $SA_LR --ot_beta $SA_BETA --trails 5 > "logs_main/Main_${ds}_${method}.txt" 2>&1
            
        elif [ "$method" == "uniprompt" ]; then
            python -u main.py --dataset $ds --method $method --tau $UNI_TAU --k $UNI_K --down_lr $UNI_LR --trails 5 > "logs_main/Main_${ds}_${method}.txt" 2>&1
            
        else
            # fine_tune 使用默认
            python -u main.py --dataset $ds --method $method --trails 5 > "logs_main/Main_${ds}_${method}.txt" 2>&1
        fi
        
        # 抓取结果
        res=$(tail -n 15 "logs_main/Main_${ds}_${method}.txt" | grep "^Accuracy" | awk '{print $2 " ± " $4}')
        if [ -z "$res" ]; then
            echo "❌ Failed (Check logs)"
        else
            echo "✅ $res"
        fi
    done
done
echo -e "\n🎉 ALL TASKS COMPLETED."