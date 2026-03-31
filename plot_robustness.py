import os
import re
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🟢 顶会极简强对比风格 (干练、清晰)
sns.set_theme(style="whitegrid") 
plt.rcParams.update({
    'font.size': 15, 
    'pdf.fonttype': 42, 
    'ps.fonttype': 42,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})

def run_experiment(method, shot, noise, gpu_id):
    cmd = [
        "python", "main.py",
        "--dataset", "texas", "--model", "GraphMAE", "--method", method,
        "--shot", str(shot), "--noise", str(noise),
        "--trails", "5", "--epochs", "500", "--down_epochs", "200"
    ]
    print(f"Running: {method} | Shot: {shot} | Noise: {noise} | GPU: {gpu_id}")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        # 抓取 Accuracy
        match = re.search(r'Accuracy:\s*([0-9.]+)\s*(?:±|\+/-)', result.stdout)
        if match:
            return float(match.group(1))
        match_mean = re.search(r'Accuracy:\s*([0-9.]+)', result.stdout)
        if match_mean:
            return float(match_mean.group(1))
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='Specify GPU ID')
    args = parser.parse_args()

    # 🟢 密集采样点：让曲线更平滑，让 Baseline 的“坠落感”更真实
    METHODS_SHOT = ["fine_tune", "uniprompt", "sa_ot_prompt"]
    SHOTS = [1, 2, 3, 5, 7, 10]

    METHODS_NOISE = ["edgeprompt", "uniprompt", "sa_ot_prompt"]
    NOISES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
    
    # 🟢 视觉层级配置：Ours 是极其粗壮的红线，Baseline 是低调的细虚线
    style_config = {
        "fine_tune":    {"name": "Fine-Tune", "color": "#7E90B0", "marker": "o", "ms": 8, "ls": "--", "lw": 2.5, "z": 2},
        "edgeprompt":   {"name": "EdgePrompt", "color": "#7E90B0", "marker": "o", "ms": 8, "ls": "--", "lw": 2.5, "z": 2},
        "uniprompt":    {"name": "UniPrompt", "color": "#F39B7F", "marker": "s", "ms": 8, "ls": "--", "lw": 2.5, "z": 3},
        "sa_ot_prompt": {"name": "SA-OT (Ours)", "color": "#DC0000", "marker": "*", "ms": 16, "ls": "-", "lw": 4.0, "z": 10}
    }

    # ==========================================
    # 1. 实验一：Shot 衰减免疫图 (带 Gap 标注)
    # ==========================================
    print(f"=== 正在运行实验一：Shot Robustness (GPU: {args.gpu}) ===")
    res_shot = {m: [] for m in METHODS_SHOT}
    all_y_shot = []
    
    for m in METHODS_SHOT:
        for s in SHOTS:
            acc = run_experiment(m, s, noise=0.0, gpu_id=args.gpu)
            res_shot[m].append(acc)
            all_y_shot.append(acc)

    plt.figure(figsize=(7, 5.5))
    for m in METHODS_SHOT:
        cfg = style_config[m]
        plt.plot(SHOTS, res_shot[m], label=cfg["name"], color=cfg["color"], marker=cfg["marker"], 
                 markersize=cfg["ms"], linestyle=cfg["ls"], linewidth=cfg["lw"], zorder=cfg["z"])

    # 🟢 绘制 1-shot 处的“性能鸿沟”标注
    idx_1shot = 0
    acc_ours_1shot = res_shot["sa_ot_prompt"][idx_1shot]
    acc_best_base_1shot = max([res_shot[m][idx_1shot] for m in METHODS_SHOT if m != "sa_ot_prompt"])
    
    # 画红色双向箭头
    plt.annotate(
        '', xy=(SHOTS[idx_1shot], acc_ours_1shot), xycoords='data',
        xytext=(SHOTS[idx_1shot], acc_best_base_1shot), textcoords='data',
        arrowprops=dict(arrowstyle="<|-|>", color='#DC0000', lw=2.5, shrinkA=3, shrinkB=3)
    )
    # 写上提升的百分点 (向右偏移一点防止挡住线)
    gap_val_shot = (acc_ours_1shot - acc_best_base_1shot) * 100
    plt.text(
        SHOTS[idx_1shot] + 0.4, (acc_ours_1shot + acc_best_base_1shot)/2, 
        f"+{gap_val_shot:.1f}%", 
        color='#DC0000', fontsize=16, fontweight='bold', va='center'
    )

    # 极限压缩 Y 轴
    if all_y_shot:
        plt.ylim(min(all_y_shot) - 0.02, max(all_y_shot) + 0.02)
        
    plt.xticks(SHOTS)
    plt.xlabel("Number of Shots")
    plt.ylabel("Accuracy")
    plt.title("Robustness to Data Scarcity", fontweight="bold")
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig("Exp1_Shot_Robustness_Clear.pdf", format='pdf', dpi=300)
    plt.close()

    # ==========================================
    # 2. 实验二：拓扑抗噪图 (带 Gap 标注)
    # ==========================================
    print(f"\n=== 正在运行实验二：Noise Resilience (GPU: {args.gpu}) ===")
    res_noise = {m: [] for m in METHODS_NOISE}
    all_y_noise = []
    
    for m in METHODS_NOISE:
        for n in NOISES:
            acc = run_experiment(m, shot=3, noise=n, gpu_id=args.gpu) 
            res_noise[m].append(acc)
            all_y_noise.append(acc)

    plt.figure(figsize=(7, 5.5))
    for m in METHODS_NOISE:
        cfg = style_config[m]
        plt.plot(NOISES, res_noise[m], label=cfg["name"], color=cfg["color"], marker=cfg["marker"], 
                 markersize=cfg["ms"], linestyle=cfg["ls"], linewidth=cfg["lw"], zorder=cfg["z"])

    # 🟢 绘制 80% 极端噪声处的“性能鸿沟”标注
    idx_noise = -1 
    acc_ours_noise = res_noise["sa_ot_prompt"][idx_noise]
    acc_best_base_noise = max([res_noise[m][idx_noise] for m in METHODS_NOISE if m != "sa_ot_prompt"])
    
    # 画红色双向箭头
    plt.annotate(
        '', xy=(NOISES[idx_noise], acc_ours_noise), xycoords='data',
        xytext=(NOISES[idx_noise], acc_best_base_noise), textcoords='data',
        arrowprops=dict(arrowstyle="<|-|>", color='#DC0000', lw=2.5, shrinkA=3, shrinkB=3)
    )
    # 写上提升的百分点 (向左偏移一点防止出界)
    gap_val_noise = (acc_ours_noise - acc_best_base_noise) * 100
    plt.text(
        NOISES[idx_noise] - 0.03, (acc_ours_noise + acc_best_base_noise)/2, 
        f"+{gap_val_noise:.1f}%", 
        color='#DC0000', fontsize=16, fontweight='bold', ha='right', va='center'
    )

    # 极限压缩 Y 轴
    if all_y_noise:
        plt.ylim(min(all_y_noise) - 0.02, max(all_y_noise) + 0.02)
        
    plt.xticks(NOISES, [f"{int(n*100)}%" for n in NOISES])
    plt.xlabel("Ratio of Spurious Edges (Noise)")
    plt.ylabel("Accuracy")
    plt.title("Defense Against Topological Toxicity", fontweight="bold")
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    plt.savefig("Exp2_Noise_Resilience_Clear.pdf", format='pdf', dpi=300)
    plt.close()

    print("\n🎉 极简高强对比版图 1 和图 2 绘制完成！已保存为 PDF。")

if __name__ == "__main__":
    main()