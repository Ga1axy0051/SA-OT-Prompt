#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

datasets=("cora" "citeseer" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")
methods=("fine_tune" "uniprompt" "sa_ot_prompt")

# 🛡️ 绝对公平的底座装甲 (5-shot 模式)
SHOTS=5
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005
TRAILS=30  # 5-shot 方差小，先跑 10 次看大盘趋势！

echo "=========================================================="
echo "🎯 THE 5-SHOT SHOWDOWN: Unleashing the True Power of OT"
echo "Common Armor: ${EPOCHS} Epochs | ${SHOTS}-Shot | Trails=${TRAILS}"
echo "=========================================================="

mkdir -p logs_5shot
touch logs_5shot/00_5SHOT_TABLE.txt

for ds in "${datasets[@]}"; do
    echo -e "\n=========================================================="
    echo "🏟️  Stadium: $ds (5-Shot)"
    
    # 🧠 注入 SA-OT 的传国玉玺参数 (继承自 1-shot 的优良基因)
    case $ds in
        "cora")       SA_TAU=0.99;  SA_K=5;  SA_LR=0.001;  SA_BETA=0.005 ;;
        "citeseer")   SA_TAU=0.85; SA_K=80; SA_LR=0.0005; SA_BETA=0.0005 ;;
        "pubmed")     SA_TAU=0.95;  SA_K=1;   SA_LR=0.005; SA_BETA=0.0005 ;;
        "cornell")    SA_TAU=0.0;  SA_K=40;  SA_LR=0.01;   SA_BETA=0.001 ;;
        "texas")      SA_TAU=0.0;  SA_K=26;  SA_LR=0.009;  SA_BETA=0.0005 ;;
        "wisconsin")  SA_TAU=0.0;  SA_K=70;  SA_LR=0.008;   SA_BETA=0.0005 ;;
        "chameleon")  SA_TAU=0.0;  SA_K=40;  SA_LR=0.015;   SA_BETA=0.0005 ;;
        "squirrel")   SA_TAU=0.0;  SA_K=55;  SA_LR=0.009;   SA_BETA=0.005  ;;
    esac

    # 🛡️ 注入 UniPrompt 官方刷榜参数
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

    echo "[$ds]" >> logs_5shot/00_5SHOT_TABLE.txt

    for method in "${methods[@]}"; do
        echo -n "   -> Running $method ... "
        log_file="logs_5shot/5shot_${ds}_${method}.txt"

        # 🚀 防 OOM 死磕循环
        while true; do
            # 断点续传检查
            if [ -f "$log_file" ] && grep -q "^Accuracy" <(tail -n 15 "$log_file"); then
                res=$(tail -n 15 "$log_file" | grep "^Accuracy" | awk '{print $2 " ± " $4}')
                echo "✅ [Cached] $res"
                echo "  $method: $res" >> logs_5shot/00_5SHOT_TABLE.txt
                break
            fi

            # 🎯 注意这里！全部改成了 --shot $SHOTS 🎯
            if [ "$method" == "sa_ot_prompt" ]; then
                python -u main.py --dataset $ds --method $method --shot $SHOTS --tau $SA_TAU --k $SA_K --down_lr $SA_LR --ot_beta $SA_BETA --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM --trails $TRAILS > "$log_file" 2>&1
            elif [ "$method" == "uniprompt" ]; then
                python -u main.py --dataset $ds --method $method --shot $SHOTS --tau $UNI_TAU --k $UNI_K --down_lr $UNI_LR --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM --trails $TRAILS > "$log_file" 2>&1
            else
                python -u main.py --dataset $ds --method $method --shot $SHOTS --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM --trails $TRAILS > "$log_file" 2>&1
            fi
            
            # 结果检查
            res=$(tail -n 15 "$log_file" | grep "^Accuracy" | awk '{print $2 " ± " $4}')
            if [ -n "$res" ]; then
                echo "✅ $res"
                echo "  $method: $res" >> logs_5shot/00_5SHOT_TABLE.txt
                break
            else
                echo -n "⚠️ OOM! Waiting 10s... "
                sleep 10
            fi
        done
    done
done
echo -e "\n=========================================================="
echo "🎉 5-SHOT TASKS COMPLETED. Check logs_5shot/00_5SHOT_TABLE.txt"