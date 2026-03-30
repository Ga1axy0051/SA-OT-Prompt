import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 🚑 OOM 绝地武士突围引擎
# ==========================================
log_dir = "/home/guoquanjiang/WXY/SA-OT-Prompt/logs_sa_ot_fine"
file_pattern = re.compile(r"([a-z]+)_(\d+)s_lr([0-9.]+)_wd([0-9e.-]+)_tau([0-9.]+)_k(\d+)_beta([0-9.]+)_p(\d+)\.txt")

GPUS = ['3']
# 🚨 极度安全的并发数，专治 k=100 时的 OOM
MAX_CONCURRENT_PER_GPU = 1  

PRETRAIN = "GraphMAE"
EPOCHS = 2000
PATIENCE = 20
TRAILS = 30

oom_tasks = []

print("🔍 正在全盘扫描 OOM 阵亡名单...")
if not os.path.exists(log_dir):
    print(f"❌ 找不到日志文件夹: {log_dir}")
    exit()

for filename in os.listdir(log_dir):
    if not filename.endswith(".txt"): 
        continue
    filepath = os.path.join(log_dir, filename)
    
    is_oom = False
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # 如果日志里没有跑出 Accuracy，或者明确报了 OOM，统统抓出来重跑！
            if "Accuracy:" not in content or "out of memory" in content.lower():
                is_oom = True
    except Exception as e:
        pass
            
    if is_oom:
        match = file_pattern.match(filename)
        if match:
            ds, shot, lr, wd, tau, k, beta, prompts = match.groups()
            oom_tasks.append((ds, int(shot), float(lr), float(wd), float(tau), int(k), float(beta), int(prompts), filepath))

print(f"🚨 警报：抓取到 {len(oom_tasks)} 个 OOM 崩溃的配置！\n")

def run_oom_task(task_info):
    ds, shot, lr, wd, tau, k, beta, prompts, old_log_file, gpu_id = task_info
    
    # 🚨 极其关键：删掉旧的报错日志，防止后面的调度器读到假缓存
    if os.path.exists(old_log_file):
        os.remove(old_log_file)
        
    cmd = [
        "python", "-u", "main.py",
        "--dataset", ds,
        "--method", "sa_ot_prompt",
        "--model", PRETRAIN,
        "--shot", str(shot),
        "--down_lr", str(lr),
        "--down_wd", str(wd),
        "--tau", str(tau),
        "--k", str(k),
        "--ot_beta", str(beta),
        "--num_prompts", str(prompts),
        "--epochs", str(EPOCHS),
        "--patience", str(PATIENCE),
        "--trails", str(TRAILS)
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    try:
        with open(old_log_file, "w") as f:
            process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
        
        with open(old_log_file, "r", encoding="utf-8", errors='ignore') as f:
            content = f.read()
            acc_matches = re.findall(r'Accuracy:\s*([0-9.]+)\s*±\s*([0-9.]+)', content)
            if acc_matches:
                return ds, shot, float(acc_matches[-1][0]), True
            else:
                return ds, shot, None, False
    except Exception as e:
        return ds, shot, None, False

if len(oom_tasks) > 0:
    print(f"🚀 开启降频突围模式！并发数降至 {MAX_CONCURRENT_PER_GPU}，确保显存充裕...")
    tasks_with_gpu = []
    for i, task in enumerate(oom_tasks):
        gpu_id = GPUS[i % len(GPUS)]
        tasks_with_gpu.append((*task, gpu_id))
        
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PER_GPU) as executor:
        future_to_task = {executor.submit(run_oom_task, task): task for task in tasks_with_gpu}
        for future in as_completed(future_to_task):
            ds, shot, acc, success = future.result()
            completed += 1
            if success:
                print(f"[{completed}/{len(oom_tasks)}] ✅ 突围成功！{ds} | {shot}s | Acc: {acc:.4f}")
            else:
                print(f"[{completed}/{len(oom_tasks)}] ❌ 依然失败！{ds} | {shot}s (这块骨头太硬，可能是 k 设得实在太大了)")
                
    print("\n🎉 OOM 拯救行动结束！请运行 recover_sota.py 更新你的终极排行榜！")
else:
    print("✅ 没有发现任何 OOM 任务，你的日志文件夹干干净净，完美收官！")