import os
import re
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 学术绘图风格设置
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def run_experiment(method, shot, noise, gpu_id):
    cmd = [
        "python", "main.py",  # 🟢 唯一致命修改：改回 main.py！
        "--dataset", "texas", "--model", "GraphMAE", "--method", method,
        "--shot", str(shot), "--noise", str(noise),
        "--trails", "5", "--epochs", "500", "--down_epochs", "200"
    ]
    print(f"Running: {method} | Shot: {shot} | Noise: {noise} | GPU: {gpu_id}")
    
    # 环境变量控制：精准锁卡
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        # 解析最后一行输出的 Accuracy
        match = re.search(r'Accuracy:\s*([0-9.]+)\s*±', result.stdout)
        if match:
            return float(match.group(1))
    except subprocess.CalledProcessError as e:
        # 🟢 如果再出错，把真实的报错原因打印出来，而不是只报一个 status 2
        print(f"❌ {method} 运行失败！")
        print(f"👉 详细报错: \n{e.stderr}\n")
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='Specify GPU ID (e.g., 2 or 5)')
    args = parser.parse_args()

    METHODS_SHOT = ["fine_tune", "uniprompt", "sa_ot_prompt"]
    SHOTS = [1, 3, 5, 10]

    METHODS_NOISE = ["edgeprompt", "uniprompt", "sa_ot_prompt"]
    NOISES = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    markers = ['o', 's', '*']
    colors = ['#E64B35', '#4DBBD5', '#00A087']

    # ==========================================
    # 1. 绘制 Shot 衰减免疫折线图
    # ==========================================
    print(f"=== 正在运行实验一：Shot Robustness (GPU: {args.gpu}) ===")
    results_shot = {m: [] for m in METHODS_SHOT}
    for m in METHODS_SHOT:
        for s in SHOTS:
            acc = run_experiment(m, s, noise=0.0, gpu_id=args.gpu)
            results_shot[m].append(acc)

    plt.figure(figsize=(7, 5))
    for idx, m in enumerate(METHODS_SHOT):
        label_name = "SA-OT (Ours)" if m == "sa_ot_prompt" else m.replace("_", "-").title()
        plt.plot(SHOTS, results_shot[m], marker=markers[idx], markersize=10, linewidth=2.5, label=label_name, color=colors[idx])

    plt.xticks(SHOTS)
    plt.xlabel("Number of Shots")
    plt.ylabel("Accuracy")
    plt.title("Immunity to Scarcity on Texas")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Exp1_Shot_Robustness.pdf", format='pdf', dpi=300)
    plt.close()

    # ==========================================
    # 2. 绘制 拓扑抗噪能力折线图
    # ==========================================
    print(f"\n=== 正在运行实验二：Noise Resilience (GPU: {args.gpu}) ===")
    results_noise = {m: [] for m in METHODS_NOISE}
    for m in METHODS_NOISE:
        for n in NOISES:
            acc = run_experiment(m, shot=3, noise=n, gpu_id=args.gpu) 
            results_noise[m].append(acc)

    plt.figure(figsize=(7, 5))
    for idx, m in enumerate(METHODS_NOISE):
        label_name = "SA-OT (Ours)" if m == "sa_ot_prompt" else m.replace("_", "-").title()
        plt.plot(NOISES, results_noise[m], marker=markers[idx], markersize=10, linewidth=2.5, label=label_name, color=colors[idx])

    plt.xticks(NOISES, [f"{int(n*100)}%" for n in NOISES])
    plt.xlabel("Ratio of Spurious Edges (Noise)")
    plt.ylabel("Accuracy")
    plt.title("Defense Against Topological Toxicity (Texas)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Exp2_Noise_Resilience.pdf", format='pdf', dpi=300)
    plt.close()

    print("\n🎉 图 1 和 图 2 绘制完成！已保存为 PDF。")

if __name__ == "__main__":
    main()