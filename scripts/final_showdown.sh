#!/bin/bash
# 👑 锁定咱们那张拥有 16G 显存的宝地
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

datasets=("cora" "citeseer" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")
methods=("fine_tune" "uniprompt" "sa_ot_prompt")

# 🛡️ 绝对公平的底座神装 (双方强制对齐)
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005
TRAILS=30

echo "=========================================================="
echo "🏆 THE FINAL SHOWDOWN: NIPS MAIN TABLE GENERATOR"
echo "Common Armor: ${EPOCHS} Epochs | WD=${DOWN_WD} | Hid=${HID_DIM}"
echo "=========================================================="

mkdir -p logs_final
touch logs_final/00_FINAL_TABLE.txt

for ds in "${datasets[@]}"; do
    echo -e "\n=========================================================="
    echo "🏟️  Stadium: $ds "
    
    # ---------------------------------------------------------
    # 🧠 1. 注入 SA-OT (Ours) 的传国玉玺参数
    # ---------------------------------------------------------
    case $ds in
        "cora")       SA_TAU=0.99;  SA_K=5;  SA_LR=0.001;  SA_BETA=0.005 ;;
        "citeseer")   SA_TAU=0.85; SA_K=80; SA_LR=0.0005; SA_BETA=0.0005 ;;
        "pubmed")     SA_TAU=0.95;  SA_K=1;   SA_LR=0.005; SA_BETA=0.0005 ;;
        "cornell")    SA_TAU=0.0;  SA_K=40;  SA_LR=0.01;   SA_BETA=0.001 ;;
        "texas")      SA_TAU=0.0;  SA_K=26;  SA_LR=0.009;  SA_BETA=0.0005 ;;
        "wisconsin")  SA_TAU=0.0;  SA_K=70;  SA_LR=0.008;   SA_BETA=0.0005 ;;
        "chameleon")  SA_TAU=0.0;  SA_K=40;  SA_LR=0.015;   SA_BETA=0.001 ;;
        "squirrel")   SA_TAU=0.0;  SA_K=55;  SA_LR=0.009;   SA_BETA=0.005  ;;
    esac

    # ---------------------------------------------------------
    # 🛡️ 2. 注入 UniPrompt 官方刷榜参数 (摘自官方代码库)
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

    echo "[$ds]" >> logs_final/00_FINAL_TABLE.txt

    for method in "${methods[@]}"; do
        echo -n "   -> Running $method ... "
        
        log_file="logs_final/Final_${ds}_${method}.txt"

        if [ "$method" == "sa_ot_prompt" ]; then
            python -u main.py --dataset $ds --method $method --tau $SA_TAU --k $SA_K --down_lr $SA_LR --ot_beta $SA_BETA --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM --trails $TRAILS > "$log_file" 2>&1
            
        elif [ "$method" == "uniprompt" ]; then
            python -u main.py --dataset $ds --method $method --tau $UNI_TAU --k $UNI_K --down_lr $UNI_LR --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM --trails $TRAILS > "$log_file" 2>&1
            
        else
            # fine_tune 作为基座参考，同样穿上神装
            python -u main.py --dataset $ds --method $method --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM --trails $TRAILS > "$log_file" 2>&1
        fi
        
        # 抓取结果
        res=$(tail -n 15 "$log_file" | grep "^Accuracy" | awk '{print $2 " ± " $4}')
        if [ -z "$res" ]; then
            echo "❌ Failed (Check logs)"
            echo "  $method: Failed" >> logs_final/00_FINAL_TABLE.txt
        else
            echo "✅ $res"
            echo "  $method: $res" >> logs_final/00_FINAL_TABLE.txt
        fi
    done
done
echo -e "\n=========================================================="
echo "🎉 ALL TASKS COMPLETED. Check logs_final/00_FINAL_TABLE.txt"