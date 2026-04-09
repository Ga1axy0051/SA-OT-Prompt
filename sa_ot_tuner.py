import os
import subprocess
import itertools
import time
import re
import json
import threading
import queue
import random

# ==========================================
# 👑 SA-OT-PROMPT ULTIMATE TUNER (V2 动态耐心排队版)
# ==========================================
GPUS = ['6'] 
MAX_CONCURRENT_PER_GPU = 3  # 允许高并发探底，配合耐心等位，4并发也是安全的

PRETRAIN = "GraphMAE2"
TRAILS = 30
EPOCHS = 2000
PATIENCE = 100
WDS = [5e-5]

# ==========================================
# 🎯 V2 第二轮：中继平滑过渡网格 (Gradual Narrowing)
# 核心策略：填补粗网格真空，稳健探索边界
# ==========================================

SPECIFIC_GRIDS = {
    # ================= 强异配图：视野依然保持开放 =================
    # Texas, Wisconsin 1s/5s, Chameleon 1s, Actor 1s 的 K 都在 120。
    # 策略：向左补 80，向右探 160。
    'wisconsin': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.05, 0.1]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.05, 0.1]}
    },
    'texas': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.0005, 0.001, 0.005]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.05, 0.1]}
    },
    'chameleon': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.005, 0.01, 0.05], 'beta': [0.0005, 0.001, 0.005]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.0005, 0.001, 0.005]}
    },
    'actor': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.01, 0.05, 0.08], 'beta': [0.005, 0.01, 0.05]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.005, 0.01, 0.05], 'beta': [0.005, 0.01, 0.05]}
    },

    # ================= 异配回调图：视野已经见顶 =================
    # Cornell 5s 的 K 最优在 60，Squirrel 在 20。
    # 策略：以它们为中心，探索周围的合理区间。
    'cornell': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.05, 0.1]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [40, 60, 80], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.05, 0.1]}
    },
    'squirrel': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [10, 20, 40], 'lr': [0.005, 0.01, 0.05], 'beta': [0.05, 0.1]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [10, 20, 40], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.0005, 0.001, 0.005]}
    },

    # ================= 强同配图：微观世界的探索 =================
    # Cora (最爱2)，Pubmed (1s爱10，5s爱2)
    # 策略：K 值在 [1, 2, 5, 10, 20] 之间建立阶梯，绝不跨越式漏掉。
    'cora': {
        1: {'tau': [0.5, 0.8, 1.0], 'k': [1, 2, 5, 10], 'lr': [0.005, 0.01, 0.05], 'beta': [0.001, 0.005, 0.01]},
        5: {'tau': [0.5, 0.8, 1.0], 'k': [1, 2, 5, 10], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.01, 0.05, 0.1]}
    },
    'pubmed': {
        1: {'tau': [0.8, 0.9, 1.0], 'k': [5, 10, 20], 'lr': [0.01, 0.05, 0.08], 'beta': [0.0001, 0.0005]},
        5: {'tau': [0.5, 0.8, 1.0], 'k': [1, 2, 5, 10], 'lr': [0.005, 0.01, 0.05], 'beta': [0.01, 0.05, 0.1]}
    },
    'citeseer': {
        1: {'tau': [0.5, 0.8, 1.0], 'k': [30, 50, 70], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.01, 0.05, 0.1]},
        5: {'tau': [0.8, 0.9, 1.0], 'k': [30, 50, 70], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.001, 0.005, 0.01]}
    }
}

RESULT_DB = "tuning_results_db.json"
LOG_DIR = "logs_v2_coarse"
MD_FILE = "SA_OT_SOTA_Leaderboard.md"

best_results = {}

if os.path.exists(RESULT_DB):
    try:
        with open(RESULT_DB, 'r') as f:
            best_results = json.load(f)
    except: pass

def parse_accuracy(log_content):
    match = re.search(r'Accuracy:\s*([0-9.]+)\s*±\s*([0-9.]+)', log_content)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def parse_dim(log_content):
    """ 🟢 核心新增：从日志中提取总维度钢印 """
    match = re.search(r'Total Dim:\s*(\d+)', log_content)
    if match:
        return int(match.group(1))
    return None

def update_leaderboard():
    with open(MD_FILE, "w", encoding="utf-8") as f:
        f.write("# 👑 SA-OT-Prompt Ultimate SOTA\n\n")
        f.write("| Dataset | Shot | Best Acc | Std | LR | WD | Tau | k | Beta | Prompts |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        sorted_keys = sorted(best_results.keys())
        for key_str in sorted_keys:
            ds, shot = key_str.split(',')
            acc, std, lr, wd, tau, k, beta, prompts = best_results[key_str]
            f.write(f"| **{ds}** | {shot} | **{acc:.4f}** | ±{std:.4f} | {lr} | {wd} | {tau} | {k} | {beta} | {prompts} |\n")
    with open(RESULT_DB, 'w') as f:
        json.dump(best_results, f, indent=4)

# ==========================================
# 🧠 核心：自主显存探测引擎状态机 (引入动态限流)
# ==========================================
task_queue = queue.Queue()
active_counts = {g: 0 for g in GPUS}
exclusive_mode = {g: False for g in GPUS}  
gpu_full_flag = {g: False for g in GPUS}  # 🔴 新增：显卡吃撑限流阀门
active_lock = threading.Lock()
print_lock = threading.Lock()

completed_tasks = 0
total_tasks = 0

def worker(gpu_id):
    global completed_tasks
    while True:
        # 🔴 [核心改动 1]：限流等位机制。如果当前卡吃撑了，且还有任务在跑，就耐心等位。
        while True:
            with active_lock:
                is_full = gpu_full_flag[gpu_id]
                active = active_counts[gpu_id]
            if is_full and active > 0:
                time.sleep(5) # 耐心等里面的任务跑完腾出空间
            else:
                break

        try:
            task_item = task_queue.get(timeout=3)
        except queue.Empty:
            break

        # 🔴 [核心改动 2]：不再直接包场，而是记录失败次数
        task_tuple, fail_count = task_item
        req_empty = (fail_count >= 3) # 事不过三，连续 OOM 3 次才允许包场独占
        ds, shot, lr, wd, tau, k, beta, prompts = task_tuple

        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = f"{LOG_DIR}/{ds}_{shot}s_lr{lr}_tau{tau}_k{k}_b{beta}_p{prompts}.txt"

        # ==========================================
        # 🟢 V2 专属：双重缓存质检 (准确率 + 256维度对齐)
        # ==========================================
        if not req_empty and os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors='ignore') as f:
                content = f.read()
                acc, std = parse_accuracy(content)
                actual_dim = parse_dim(content)
                
                # 如果分数存在，并且维度完美匹配 256，认定为合法缓存
                if acc is not None and actual_dim == 256: 
                    with print_lock:
                        completed_tasks += 1
                        if completed_tasks % 10 == 0:
                            print(f"Progress: {completed_tasks}/{total_tasks} [Cached]")
                    task_queue.task_done()
                    continue
            
            # 走到这里说明缓存存在但失效（没有256维度或者报错残缺），直接物理销毁
            if os.path.exists(log_file):
                try:
                    os.remove(log_file)
                except:
                    pass

        # ---------------- 🛡️ 智能准入与错峰调度系统 ----------------
        can_run = False
        with active_lock:
            if exclusive_mode[gpu_id] or active_counts[gpu_id] >= MAX_CONCURRENT_PER_GPU:
                can_run = False
            elif req_empty:
                if active_counts[gpu_id] > 0:
                    can_run = False  # 要求包场，但还有人没跑完，等一下
                else:
                    can_run = True
                    active_counts[gpu_id] += 1
                    exclusive_mode[gpu_id] = True # 开启包场锁死
            else:
                can_run = True
                active_counts[gpu_id] += 1

        if not can_run:
            # 进不去显卡，重新排到队尾！让后面的小任务插队！
            task_queue.put(task_item)
            time.sleep(random.uniform(2, 5)) # 随机延迟防死锁
            task_queue.task_done()
            continue

        # 🚀 [终极防抖] 错峰发车，防止 data.to(device) 瞬间爆炸
        if not req_empty:
            wait_time = active_counts[gpu_id] * 4 
            time.sleep(wait_time) 

        # ---------------- 🛡️ 严格锁死运行命令 ----------------
        cmd = [
            "python", "-u", "main.py", "--dataset", ds, "--method", "sa_ot_prompt", "--model", PRETRAIN,
            "--shot", str(shot), 
            "--down_lr", str(lr), 
            "--clf_lr", "0.05",         # 🔴 分类器直接锁死 0.05
            "--down_wd", str(wd), 
            "--tau", str(tau), 
            "--k", str(k), 
            "--ot_beta", str(beta),
            "--num_prompts", str(prompts), 
            "--epochs", "2000",         
            "--down_epochs", "2000",    # 💣 彻底放开下游微调上限
            "--patience", "100",        # 🟡 耐心给到 100
            "--trails", str(TRAILS), 
            "--hid_dim", "256"          # 🟢 维度锁死
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        try:
            # 正常启动子进程
            process = subprocess.run(cmd, env=env, capture_output=True, text=True)
            output = process.stdout + process.stderr
            
            with open(log_file, "w") as f:
                f.write(output)

            # 🧠 全量小写化匹配
            output_lower = output.lower()

            # 🧠 增强型 OOM 探测网
            is_oom = (
                "out of memory" in output_lower or 
                "oom" in output_lower or 
                "allocation on device" in output_lower or
                process.returncode == 137 
            )
            
            # 🔴 [核心改动 3]：精准的状态标记与限流
            with active_lock:
                active_counts[gpu_id] -= 1
                if req_empty:
                    exclusive_mode[gpu_id] = False 

                if is_oom:
                    # 如果包场还炸，或者唯一在跑也炸，说明物理显存装不下
                    if req_empty or active_counts[gpu_id] == 0:
                        fatal_oom = True
                    else:
                        fatal_oom = False
                        gpu_full_flag[gpu_id] = True # 📢 广播限流：这张卡吃撑了，门外排队！
                else:
                    fatal_oom = False
                    if process.returncode == 0:
                        gpu_full_flag[gpu_id] = False # 📢 广播放行：跑成功腾出空间了，可以进新任务了！

            if is_oom:
                if fatal_oom:
                    with print_lock:
                        completed_tasks += 1
                        print(f"{completed_tasks}/{total_tasks} 💀 [FATAL OOM] {ds} {shot}s (Exceeds Physical GPU Memory!)")
                    task_queue.task_done()
                else:
                    with print_lock:
                        print(f"⏳ [GPU FULL] {ds} {shot}s - Squeezed out. Retrying normally later (Fail count: {fail_count+1})")
                    # 作为普通任务重新排队，失败次数 +1
                    task_queue.put((task_tuple, fail_count + 1))
                    task_queue.task_done()
                continue

            # 检查进程是否正常退出
            if process.returncode != 0:
                acc, std = None, None
            else:
                acc, std = parse_accuracy(output)
            
        except Exception as e:
            with active_lock:
                active_counts[gpu_id] -= 1
                if req_empty:
                    exclusive_mode[gpu_id] = False
            with open(log_file, "a") as f:
                f.write(f"\n[CRASH ERROR]: {str(e)}\n")
            acc, std = None, None

        # ---------------- 📊 成绩写入 ----------------
        with print_lock:
            completed_tasks += 1
            if acc is not None:
                key = f"{ds},{shot}"
                if key not in best_results or acc > best_results[key][0]:
                    print(f"🔥 NEW SOTA for {ds.upper()} {shot}s! Acc: {acc:.4f} (lr={lr}, tau={tau}, k={k}, beta={beta})")
                    best_results[key] = [acc, std, lr, wd, tau, k, beta, prompts]
                    update_leaderboard()
                elif completed_tasks % 10 == 0:
                    print(f"Progress: {completed_tasks}/{total_tasks} explored...")
            else:
                print(f"Progress: {completed_tasks}/{total_tasks} [ERROR/CRASH] {ds} {shot}s")

        task_queue.task_done()

def main():
    global total_tasks
    print("🚀 SA-OT-PROMPT BOMBING INITIATED (Dynamic Backpressure & Dimension Guard)")
    print(f"GPUs: {GPUS} | Workers/GPU: {MAX_CONCURRENT_PER_GPU}")
    
    all_tasks = []
    for ds in SPECIFIC_GRIDS.keys():
        for shot in [1, 3, 5]:
            if shot not in SPECIFIC_GRIDS[ds]:
                continue
            grid = SPECIFIC_GRIDS[ds][shot]
            current_prompts = [10]
            combos = list(itertools.product([ds], [shot], grid['lr'], WDS, grid['tau'], grid['k'], grid['beta'], current_prompts))
            all_tasks.extend(combos)
    
    total_tasks = len(all_tasks)
    print(f"🎯 Total Configs: {total_tasks}")
    
    # 🔀 核心动作：打乱任务，大图小图混排，让缝隙插队最大化！
    random.seed(42)
    random.shuffle(all_tasks)
    print("🔀 Tasks shuffled for optimal VRAM packing!\n")
    
    # 🚀 绝对纯净的入队：没有任何经验预判，初始 fail_count 为 0
    for t in all_tasks:
        task_queue.put((t, 0))

    threads = []
    for i in range(len(GPUS) * MAX_CONCURRENT_PER_GPU):
        gpu_id = GPUS[i % len(GPUS)]
        t = threading.Thread(target=worker, args=(gpu_id,))
        time.sleep(1.5) # 给线程启动本身加个微小间隔
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()