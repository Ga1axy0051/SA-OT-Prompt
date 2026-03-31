import os
import re

# 你的日志文件夹路径
log_dir = "/home/guoquanjiang/WXY/SA-OT-Prompt/logs_sa_ot_fine"

# 用于匹配文件名的正则表达式 (完美对应咱们之前的命名规则)
# 格式: dataset_shots_lrlr_wdwd_tautau_kk_betabeta_pprompts.txt
file_pattern = re.compile(r"([a-z]+)_(\d+)s_lr([0-9.]+)_wd([0-9e.-]+)_tau([0-9.]+)_k(\d+)_beta([0-9.]+)_p(\d+)\.txt")

best_results = {}

print("🔍 正在全盘扫描日志文件夹，寻找遗失的 SOTA...")

if not os.path.exists(log_dir):
    print(f"❌ 找不到文件夹: {log_dir}，请检查路径是否正确！")
    exit()

for filename in os.listdir(log_dir):
    if not filename.endswith(".txt"):
        continue

    match = file_pattern.match(filename)
    if not match:
        continue

    ds, shot, lr, wd, tau, k, beta, prompts = match.groups()
    shot = int(shot)

    filepath = os.path.join(log_dir, filename)
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # 找到日志里最后出现的 Accuracy
            acc_matches = re.findall(r'Accuracy:\s*([0-9.]+)\s*±\s*([0-9.]+)', content)
            if acc_matches:
                acc, std = float(acc_matches[-1][0]), float(acc_matches[-1][1])

                key = (ds, shot)
                # 更新最高分
                if key not in best_results or acc > best_results[key][0]:
                    best_results[key] = (acc, std, lr, wd, tau, k, beta, prompts)
    except Exception as e:
        print(f"读取文件 {filename} 时出错: {e}")

print("\n🎉 恢复成功！这是你遗失的巅峰战报：\n")

# 直接打印 Markdown 格式的表格，方便你复制回文档
print("| Dataset | Shot | Best Acc | Std | LR | WD | Tau | k | Beta | Prompts |")
print("|---|---|---|---|---|---|---|---|---|---|")

# 按数据集和 Shot 排序打印
for key in sorted(best_results.keys(), key=lambda x: (x[0], x[1])):
    ds, shot = key
    acc, std, lr, wd, tau, k, beta, prompts = best_results[key]
    print(f"| **{ds}** | {shot} | **{acc:.4f}** | ±{std:.4f} | {lr} | {wd} | {tau} | {k} | {beta} | {prompts} |")

# 顺手帮你存个文件
with open("Recovered_SA_OT_SOTA.md", "w", encoding="utf-8") as f:
    f.write("# 🛡️ Recovered SA-OT-Prompt SOTA\n\n")
    f.write("| Dataset | Shot | Best Acc | Std | LR | WD | Tau | k | Beta | Prompts |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|\n")
    for key in sorted(best_results.keys(), key=lambda x: (x[0], x[1])):
        ds, shot = key
        acc, std, lr, wd, tau, k, beta, prompts = best_results[key]
        f.write(f"| **{ds}** | {shot} | **{acc:.4f}** | ±{std:.4f} | {lr} | {wd} | {tau} | {k} | {beta} | {prompts} |\n")

print("\n💾 表格已同时保存到当前目录下的 'Recovered_SA_OT_SOTA.md' 文件中！")