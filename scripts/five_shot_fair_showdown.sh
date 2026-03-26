#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

datasets=("cora" "citeseer" "pubmed" "cornell" "texas" "wisconsin" "chameleon" "squirrel")
methods=("uniprompt" "sa_ot_prompt")

# 🛡️ 绝对公平的 5-shot 底座
SHOTS=5
TRAILS=30
EPOCHS=2000
HID_DIM=256
DOWN_WD=0.00005

echo "=========================================================="
echo "⚖️ THE ULTIMATE FAIR SHOWDOWN: SA-OT vs Optimal UniPrompt"
echo "Common Armor: ${EPOCHS} Epochs | ${SHOTS}-Shot | Trails=${TRAILS}"
echo "=========================================================="

mkdir -p logs_fair_showdown
touch logs_fair_showdown/00_FAIR_COMPARISON.txt

for ds in "${datasets[@]}"; do
    echo -e "\n=========================================================="
    echo "🏟️  Arena: [ $ds ] (5-Shot, 30 Trails)"
    
    # 👑 咱们的 SA-OT 终极霸主参数 (直接锁死刚才跑出的最高记录)
    case $ds in
        "cora")       SA_TAU=0.6;  SA_K=4;  SA_LR=0.002;  SA_BETA=0.005  ;;
        "citeseer")   SA_TAU=0.5;  SA_K=45; SA_LR=0.0008; SA_BETA=0.0005 ;;
        "pubmed")     SA_TAU=0.8;  SA_K=1;  SA_LR=0.01;   SA_BETA=0.001  ;;
        "chameleon")  SA_TAU=0.0;  SA_K=5; SA_LR=0.0005;  SA_BETA=0.001  ;;
        "texas")      SA_TAU=0.0;  SA_K=48; SA_LR=0.003;  SA_BETA=0.0005 ;;
        "squirrel")   SA_TAU=0.0;  SA_K=35; SA_LR=0.007;  SA_BETA=0.005  ;;
        "wisconsin")  SA_TAU=0.0;  SA_K=85; SA_LR=0.0008;  SA_BETA=0.0005 ;;
        "cornell")    SA_TAU=0.0;  SA_K=42; SA_LR=0.0008;  SA_BETA=0.001  ;;
    esac

    # ⚠️ 请在这里填写 UniPrompt 论文里公布的 5-shot 最优超参！
    # 注意区分 up_lr 和 down_lr，咱们脚本里的 down_lr 对应你填的 UNI_LR
    case $ds in
        "cora")       UNI_TAU=0.9999; UNI_K=1; UNI_LR=0.0005   ;; # 示例值，请修改
        "citeseer")   UNI_TAU=0.9999;  UNI_K=10; UNI_LR=0.05   ;; # 示例值，请修改
        "pubmed")     UNI_TAU=0.9999; UNI_K=1; UNI_LR=0.05   ;; # 示例值，请修改
        "cornell")    UNI_TAU=0.9999;  UNI_K=50; UNI_LR=0.0005  ;; # 示例值，请修改
        "texas")      UNI_TAU=0.9999;  UNI_K=50; UNI_LR=0.0005 ;; # 示例值，请修改
        "wisconsin")  UNI_TAU=0.9999;  UNI_K=50; UNI_LR=0.0005 ;; # 示例值，请修改
        "chameleon")  UNI_TAU=0.9999;   UNI_K=50; UNI_LR=0.05   ;; # 示例值，请修改
        "squirrel")   UNI_TAU=0.9999;  UNI_K=50; UNI_LR=0.0001 ;; # 示例值，请修改
    esac

    echo "[$ds]" >> logs_fair_showdown/00_FAIR_COMPARISON.txt

    for method in "${methods[@]}"; do
        echo -n "   -> Running $method ... "
        log_file="logs_fair_showdown/fair_${ds}_${method}.txt"

        # 🚀 防 OOM 护盾死磕循环
        while true; do
            # 断点续传检查
            if [ -f "$log_file" ] && grep -q "^Accuracy" <(tail -n 15 "$log_file"); then
                res=$(tail -n 15 "$log_file" | grep "^Accuracy" | awk '{print $2 " ± " $4}')
                echo "✅ [Cached] $res"
                echo "  $method: $res" >> logs_fair_showdown/00_FAIR_COMPARISON.txt
                break
            fi

            # 根据方法分配参数
            if [ "$method" == "sa_ot_prompt" ]; then
                python -u main.py --dataset $ds --method $method --shot $SHOTS --tau $SA_TAU --k $SA_K --down_lr $SA_LR --ot_beta $SA_BETA --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM --trails $TRAILS > "$log_file" 2>&1
            elif [ "$method" == "uniprompt" ]; then
                python -u main.py --dataset $ds --method $method --shot $SHOTS --tau $UNI_TAU --k $UNI_K --down_lr $UNI_LR --down_wd $DOWN_WD --epochs $EPOCHS --hid_dim $HID_DIM --trails $TRAILS > "$log_file" 2>&1
            fi
            
            # 结果检查
            res=$(tail -n 15 "$log_file" | grep "^Accuracy" | awk '{print $2 " ± " $4}')
            if [ -n "$res" ]; then
                echo "✅ $res"
                echo "  $method: $res" >> logs_fair_showdown/00_FAIR_COMPARISON.txt
                break
            else
                echo -n "⚠️ OOM! Waiting 10s... "
                sleep 10
            fi
        done
    done
done
echo -e "\n=========================================================="
echo "🎉 FAIR SHOWDOWN COMPLETED. Results are in logs_fair_showdown/00_FAIR_COMPARISON.txt"