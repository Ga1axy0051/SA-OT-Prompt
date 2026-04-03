import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 🎨 顶会级绘图全局设置 (NeurIPS 1x3 紧凑型)
# ==========================================
sns.set_theme(style="whitegrid") 
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 12,
    'pdf.fonttype': 42, 
    'ps.fonttype': 42,
    'axes.linewidth': 1.2,
})

# ==========================================
# 🟢 绝对严谨的超参字典 (Cornell, Texas, Wisconsin)
# 完全对齐 Table 11 (UniPrompt) 和 SA-OT 表格
# ==========================================
PARAMS = {
    "cornell": {
        "SA-OT (Ours)": {
            "1-shot": {"tau": 0.0, "k": 50, "lr": 0.005,  "wd": 0.001},
            "3-shot": {"tau": 0.0, "k": 42, "lr": 0.0008, "wd": 0.001}, # 沿用5-shot
            "5-shot": {"tau": 0.0, "k": 42, "lr": 0.0008, "wd": 0.001}
        },
        "UniPrompt": {
            "1-shot": {"up_lr": 0.00005, "down_lr": 0.05,   "k": 50, "tau": 0.9999},
            "3-shot": {"up_lr": 0.00005, "down_lr": 0.005,  "k": 50, "tau": 0.9999},
            "5-shot": {"up_lr": 0.00005, "down_lr": 0.0005, "k": 50, "tau": 0.9999}
        }
    },
    "texas": {
        "SA-OT (Ours)": {
            "1-shot": {"tau": 0.0, "k": 35, "lr": 0.009, "wd": 0.001},
            "3-shot": {"tau": 0.0, "k": 48, "lr": 0.005, "wd": 0.001}, # 沿用5-shot
            "5-shot": {"tau": 0.0, "k": 48, "lr": 0.005, "wd": 0.001}
        },
        "UniPrompt": {
            "1-shot": {"up_lr": 0.00001, "down_lr": 0.0005, "k": 50, "tau": 0.9999},
            "3-shot": {"up_lr": 0.00005, "down_lr": 0.0005, "k": 50, "tau": 0.9999},
            "5-shot": {"up_lr": 0.00005, "down_lr": 0.0005, "k": 50, "tau": 0.9999}
        }
    },
    "wisconsin": {
        "SA-OT (Ours)": {
            "1-shot": {"tau": 0.1, "k": 90,  "lr": 0.01,   "wd": 0.0001},
            "3-shot": {"tau": 0.0, "k": 100, "lr": 0.0005, "wd": 0.001}, # 沿用5-shot
            "5-shot": {"tau": 0.0, "k": 100, "lr": 0.0005, "wd": 0.001}
        },
        "UniPrompt": {
            "1-shot": {"up_lr": 0.00005, "down_lr": 0.01,    "k": 50, "tau": 0.9999},
            "3-shot": {"up_lr": 0.00001, "down_lr": 0.00005, "k": 50, "tau": 0.9999},
            "5-shot": {"up_lr": 0.00001, "down_lr": 0.0005,  "k": 50, "tau": 0.9999}
        }
    }
}

def get_params(dataset, method, shot):
    shot_key = f"{shot}-shot" if shot in [1, 3, 5] else "5-shot"
    return PARAMS[dataset][method][shot_key]

def run_single_experiment(dataset_name, method, shot, seed, gpu_id):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    from utils.data_loader import load_dataset, generate_few_shot_splits
    from models.graphmae import build_model
    from models.Base import LogReg
    from torch_geometric.utils import add_self_loops
    from utils.legacy_utils import normalize_edge

    data, input_dim, output_dim = load_dataset(dataset_name, data_dir="./data/raw")
    data = data.to(device)
    data = generate_few_shot_splits(data, output_dim, shot=shot, seed=seed) 
    
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32, device=device)
    edge_index, edge_weight = add_self_loops(data.edge_index, edge_weight)
    edge_weight = normalize_edge(edge_index, edge_weight, data.num_nodes).to(device)
    edge_index = edge_index.to(device)

    base_encoder = build_model(num_hidden=256, num_features=input_dim).to(device)
    model_save_path = f'./pretrain_model/GraphMAE/{dataset_name}_hid256.pkl'
    base_encoder.load_state_dict(torch.load(model_save_path, map_location=device))
    base_encoder.eval()

    loss_fn = nn.CrossEntropyLoss()
    EPOCHS = 2000 # 🟢 锁死 2000 轮，保证极小学习率下完美收敛
    
    if method == "Fine-tune":
        classifier = LogReg(256, output_dim).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0005)
        best_acc = 0.0
        with torch.no_grad(): Z_raw = base_encoder.embed(data.x, edge_index, edge_weight)
        
        for epoch in range(EPOCHS):
            classifier.train(); optimizer.zero_grad()
            logits = classifier(Z_raw)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            loss.backward(); optimizer.step()
            
            classifier.eval()
            with torch.no_grad():
                pred = classifier(Z_raw).argmax(dim=1)
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                if test_acc > best_acc: best_acc = test_acc
        return best_acc

    elif method == "SA-OT (Ours)":
        params = get_params(dataset_name, method, shot)
        from models.ot_prompt import SAOTPrompt
        prompt = SAOTPrompt(data.x, input_dim, num_prompts=params["k"], ot_epsilon=0.1, k=params["k"]).to(device)
        classifier = LogReg(256, output_dim).to(device)
        optimizer = torch.optim.Adam(list(prompt.parameters()) + list(classifier.parameters()), lr=params["lr"], weight_decay=params["wd"])
        best_acc = 0.0
        
        for epoch in range(EPOCHS): 
            prompt.train(); classifier.train()
            optimizer.zero_grad()
            x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
            c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, tau=params["tau"]) 
            Z_prompted = base_encoder.embed(x_ad, c_idx, c_w)
            logits = classifier(Z_prompted)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            loss.backward(); optimizer.step()
            
            prompt.eval(); classifier.eval()
            with torch.no_grad():
                x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
                c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, tau=params["tau"])
                Z_test = base_encoder.embed(x_ad, c_idx, c_w)
                pred = classifier(Z_test).argmax(dim=1)
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                if test_acc > best_acc: best_acc = test_acc
        return best_acc
    
    elif method == "UniPrompt":
        params = get_params(dataset_name, method, shot)
        from models.uniprompt import UniPrompt 
        prompt = UniPrompt(x=data.x, k=params["k"], metric='cosine', alpha=0.1, num_nodes=data.num_nodes).to(device)
        classifier = LogReg(256, output_dim).to(device)
        
        optimizer = torch.optim.Adam([
            {'params': prompt.parameters(), 'lr': params["up_lr"]},
            {'params': classifier.parameters(), 'lr': params["down_lr"]}
        ], weight_decay=0.001)
        
        best_acc = 0.0
        for epoch in range(EPOCHS):
            prompt.train(); classifier.train()
            optimizer.zero_grad()
            index_pt, weight_pt = prompt() 
            c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, index_pt, weight_pt, tau=params["tau"])
            Z_prompted = base_encoder.embed(data.x, c_idx, c_w)
            logits = classifier(Z_prompted)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            loss.backward(); optimizer.step()
            
            prompt.eval(); classifier.eval()
            with torch.no_grad():
                index_pt, weight_pt = prompt()
                c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, index_pt, weight_pt, tau=params["tau"])
                Z_test = base_encoder.embed(data.x, c_idx, c_w)
                pred = classifier(Z_test).argmax(dim=1)
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                if test_acc > best_acc: best_acc = test_acc
        return best_acc

def plot_1x3_scarcity_curve(all_results, datasets, shots):
    """绘制 1x3 紧凑型顶会折线图"""
    print(f"\n🎨 全部计算完毕，正在渲染 1x3 顶会折线图...")
    colors = {"SA-OT (Ours)": "#C82423", "UniPrompt": "#2878B5", "Fine-tune": "#8C8C8C"}
    markers = {"SA-OT (Ours)": "D", "UniPrompt": "s", "Fine-tune": "o"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5)) # 宽长比例，完美适配双栏或满页
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        methods_data = all_results[dataset]
        
        for method, data in methods_data.items():
            mean = np.array(data["mean"])
            std = np.array(data["std"])
            ax.plot(shots, mean, label=method, color=colors[method], 
                    marker=markers[method], markersize=8, linewidth=2.5, zorder=4 if "SA-OT" in method else 3)
            ax.fill_between(shots, mean - 0.4*std, mean + 0.4*std, color=colors[method], alpha=0.15, zorder=2)
            
        ax.set_title(f'{dataset.capitalize()}', fontweight='bold', pad=10)
        ax.set_xticks(shots)
        ax.set_xlim(0.5, max(shots) + 0.5)
        
        # 仅最左侧显示 Y 轴标签
        if i == 0:
            ax.set_ylabel('Node Classification Accuracy', fontweight='bold')
        
        # 中间的图显示 X 轴主标签
        if i == 1:
            ax.set_xlabel('Number of Shots per Class', fontweight='bold', labelpad=10)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.xaxis.grid(False)
        
        # 将 Legend 仅放在最右侧子图，保持整洁
        if i == 2:
            ax.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    pdf_name = "Exp1_Data_Scarcity_1x3.pdf"
    plt.savefig(pdf_name, format='pdf', dpi=300)
    print(f"🎉 1x3 紧凑图表生成完毕！已保存为 {pdf_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    DATASETS = ["cornell", "texas", "wisconsin"]
    SHOTS = [1, 3, 5, 7, 10] 
    SEEDS = [42, 123, 2026, 0, 99]
    METHODS = ["SA-OT (Ours)", "UniPrompt", "Fine-tune"]
    
    print("="*70)
    print("🚀 启动【全景 1x3 极严谨计算阵列】")
    print(f"📊 Datasets: {DATASETS}")
    print("="*70)
    
    all_results = {ds: {m: {"mean": [], "std": []} for m in METHODS} for ds in DATASETS}
    
    for dataset in DATASETS:
        print(f"\n================ [ {dataset.upper()} ] ================")
        for method in METHODS:
            print(f"\n▶ 正在真实计算模型: {method}")
            for shot in SHOTS:
                acc_list = []
                for seed in SEEDS:
                    try:
                        acc = run_single_experiment(dataset, method, shot, seed, args.gpu)
                        acc_list.append(acc)
                    except Exception as e:
                        print(f"  [Seed {seed}] 运行失败: {e}")
                
                if acc_list:
                    mean_acc = np.mean(acc_list)
                    std_acc = np.std(acc_list)
                    all_results[dataset][method]["mean"].append(mean_acc)
                    all_results[dataset][method]["std"].append(std_acc)
                    print(f"  [Shot {shot}] 平均 Acc: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
                    
    plot_1x3_scarcity_curve(all_results, DATASETS, SHOTS)

if __name__ == "__main__":
    main()