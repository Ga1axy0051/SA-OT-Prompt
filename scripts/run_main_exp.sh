#!/bin/bash
datasets=("cora" "texas")
methods=("linear_probe" "fine_tune" "sa_ot_prompt")

echo "=========================================================="
echo "Generating Mini-Table 1 (GraphMAE Base)"
echo "=========================================================="

for dataset in "${datasets[@]}"; do
    echo -e "\n>>> Testing Dataset: $dataset"
    
    if [ "$dataset" == "texas" ]; then
        tau=0.9999; patience=100; k=5; down_lr=0.01 # <-- 把 k 降到 5 或者 10
    else
        tau=0.01; patience=50; k=50; down_lr=0.001
    fi
    
    for method in "${methods[@]}"; do
        echo "Running Method: $method"
        python main.py \
            --dataset $dataset \
            --method $method \
            --tau $tau \
            --patience $patience \
            --k $k \
            --down_lr $down_lr \
            --trails 3 > logs_${dataset}_${method}.txt
        
        # 提取并直接在终端打印最后几行的 Final Report，看着贼爽！
        tail -n 6 logs_${dataset}_${method}.txt
    done
done