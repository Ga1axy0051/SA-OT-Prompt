import os
import re
import pandas as pd

LOG_DIR = "logs_v2_coarse"

def parse_logs():
    data = []
    # 匹配文件名的正则，例如: cora_1s_lr0.001_tau0.95_k5_b0.01_p5.txt
    file_pattern = re.compile(r"(.+)_(\d+)s_lr([0-9.]+)_tau([0-9.]+)_k(\d+)_b([0-9.]+)_p(\d+)\.txt")
    
    if not os.path.exists(LOG_DIR):
        print(f"⚠️ 找不到目录 {LOG_DIR}")
        return pd.DataFrame()

    for filename in os.listdir(LOG_DIR):
        if not filename.endswith(".txt"):
            continue
            
        match = file_pattern.match(filename)
        if not match:
            continue
            
        ds, shot, lr, tau, k, beta, prompts = match.groups()
        filepath = os.path.join(LOG_DIR, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # 提取 Acc
                acc_match = re.search(r'Accuracy:\s*([0-9.]+)\s*±', content)
                if acc_match:
                    acc = float(acc_match.group(1))
                    data.append({
                        "dataset": ds,
                        "shot": int(shot),
                        "lr": float(lr),
                        "tau": float(tau),
                        "k": int(k),
                        "beta": float(beta),
                        "prompts": int(prompts),
                        "acc": acc
                    })
        except Exception as e:
            pass
            
    return pd.DataFrame(data)

def analyze_trends(df):
    if df.empty:
        print("没有找到有效的日志数据！")
        return
        
    datasets = df['dataset'].unique()
    shots = [1, 5]
    
    for ds in datasets:
        for shot in shots:
            subset = df[(df['dataset'] == ds) & (df['shot'] == shot)]
            if subset.empty:
                continue
                
            print("="*60)
            print(f"🚀 数据集: {ds.upper()} | Shot: {shot}")
            print("="*60)
            
            # 1. 打印全局最佳组合
            best_idx = subset['acc'].idxmax()
            best_row = subset.loc[best_idx]
            print(f"🏆 最高 Acc: {best_row['acc']:.4f}")
            print(f"👉 最佳参数: lr={best_row['lr']}, tau={best_row['tau']}, k={best_row['k']}, beta={best_row['beta']}, prompts={best_row['prompts']}\n")
            
            # 2. 核心超参趋势分析 (取均值)
            print("📈 [超参趋势分析] (展示特定参数值下的平均准确率):")
            params_to_analyze = ['tau', 'k', 'lr', 'beta', 'prompts']
            
            for param in params_to_analyze:
                # 按照该超参的值进行分组，计算平均准确率和最高准确率
                trend = subset.groupby(param)['acc'].agg(['mean', 'max']).reset_index()
                trend = trend.sort_values(by=param)
                
                trend_strs = []
                for _, row in trend.iterrows():
                    val = row[param]
                    # 如果是浮点数且没有小数部分，转为整数显示更干净
                    val_str = f"{int(val)}" if val == int(val) and param in ['k', 'prompts'] else f"{val}"
                    trend_strs.append(f"{val_str} (均:{row['mean']:.4f}/极:{row['max']:.4f})")
                
                print(f"   - {param.upper().ljust(7)}: " + " ➡️ ".join(trend_strs))
            
            # 3. 圈定 Top 10% 的安全范围
            top_10_percent = int(len(subset) * 0.1) if len(subset) >= 10 else 3
            top_configs = subset.nlargest(top_10_percent, 'acc')
            
            print(f"\n🎯 [下一轮建议搜索范围] (基于 Top {top_10_percent} 的参数分布):")
            for param in params_to_analyze:
                min_val = top_configs[param].min()
                max_val = top_configs[param].max()
                best_val = best_row[param]
                
                if param in ['k', 'prompts']:
                    print(f"   - {param.upper()}: 范围 [{int(min_val)}, {int(max_val)}] | 强推荐在 {int(best_val)} 附近")
                else:
                    print(f"   - {param.upper()}: 范围 [{min_val}, {max_val}] | 强推荐在 {best_val} 附近")
            print("\n")

if __name__ == "__main__":
    print("正在解析日志文件，请稍候...")
    df = parse_logs()
    print(f"✅ 成功提取了 {len(df)} 条运行记录！\n")
    analyze_trends(df)