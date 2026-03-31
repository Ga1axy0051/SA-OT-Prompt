import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 🎨 顶会级绘图全局设置
# ==========================================
sns.set_theme(style="white") 
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'pdf.fonttype': 42, 
    'ps.fonttype': 42,
})

# 🟢 WebKB 三剑客 1-shot 最优超参字典
HYPERPARAMS = {
    "cornell":   {"tau": 0.0, "k": 50, "lr": 0.005, "wd": 0.001},
    "texas":     {"tau": 0.0, "k": 35, "lr": 0.009, "wd": 0.001},
    "wisconsin": {"tau": 0.1, "k": 90, "lr": 0.01,  "wd": 0.0001}
}

def get_class_cosine_similarity(Z, labels, num_classes):
    """计算纯正的 Cosine Similarity 矩阵，并提取 Intra/Inter 及 Delta"""
    Z_norm = F.normalize(Z, p=2, dim=1)
    sim_matrix = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(num_classes):
            Z_i = Z_norm[labels == i]
            Z_j = Z_norm[labels == j]
            if len(Z_i) == 0 or len(Z_j) == 0:
                continue
            cos_sim = torch.mm(Z_i, Z_j.t())
            sim_matrix[i, j] = cos_sim.mean().item()
            
    # 计算类内 (Intra) 和类间 (Inter)
    intra_sim = np.trace(sim_matrix) / num_classes
    inter_sim = (np.sum(sim_matrix) - np.trace(sim_matrix)) / (num_classes * (num_classes - 1))
    delta = intra_sim - inter_sim
    
    return sim_matrix, intra_sim, inter_sim, delta

def run_single_dataset_single_seed(dataset_name, shot, seed, gpu_id, params):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    from utils.data_loader import load_dataset, generate_few_shot_splits
    from models.graphmae import build_model
    from models.ot_prompt import SAOTPrompt
    from models.Base import LogReg
    from torch_geometric.utils import add_self_loops
    from utils.legacy_utils import normalize_edge

    # 1. 数据
    data, input_dim, output_dim = load_dataset(dataset_name, data_dir="./data/raw")
    data = data.to(device)
    data = generate_few_shot_splits(data, output_dim, shot=shot, seed=seed) 
    num_classes = output_dim
    
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32, device=device)
    edge_index, edge_weight = add_self_loops(data.edge_index, edge_weight)
    edge_weight = normalize_edge(edge_index, edge_weight, data.num_nodes).to(device)
    edge_index = edge_index.to(device)

    # 2. Before (GraphMAE)
    base_encoder = build_model(num_hidden=256, num_features=input_dim).to(device)
    model_save_path = f'./pretrain_model/GraphMAE/{dataset_name}_hid256.pkl'
    base_encoder.load_state_dict(torch.load(model_save_path, map_location=device))
    base_encoder.eval()
    
    with torch.no_grad():
        Z_raw = base_encoder.embed(data.x, edge_index, edge_weight)
    sim_raw, intra_raw, inter_raw, delta_raw = get_class_cosine_similarity(Z_raw, data.y, num_classes)

    # 3. After (SA-OT)
    prompt = SAOTPrompt(data.x, input_dim, num_prompts=params["k"], ot_epsilon=0.1, k=params["k"]).to(device)
    classifier = LogReg(256, output_dim).to(device)
    optimizer = torch.optim.Adam(list(prompt.parameters()) + list(classifier.parameters()), lr=params["lr"], weight_decay=params["wd"])
    loss_fn = nn.CrossEntropyLoss()
    
    prompt.train()
    classifier.train()
    for epoch in range(200): 
        optimizer.zero_grad()
        x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
        c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, tau=params["tau"]) 
        Z_prompted = base_encoder.embed(x_ad, c_idx, c_w)
        logits = classifier(Z_prompted)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    prompt.eval()
    base_encoder.eval()
    with torch.no_grad():
        x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
        c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, tau=params["tau"])
        Z_prompted = base_encoder.embed(x_ad, c_idx, c_w)
        
    sim_prompted, intra_p, inter_p, delta_p = get_class_cosine_similarity(Z_prompted, data.y, num_classes)
    
    return sim_raw, intra_raw, inter_raw, delta_raw, sim_prompted, intra_p, inter_p, delta_p, num_classes

def run_dataset_with_seeds(dataset_name, shot, gpu_id):
    SEEDS = [42, 123, 2026, 0, 99] 
    params = HYPERPARAMS[dataset_name]
    print(f"\n⚡ 正在处理: {dataset_name.upper()} (Seeds: {len(SEEDS)})")
    
    all_sim_r, all_in_r, all_out_r, all_d_r = [], [], [], []
    all_sim_p, all_in_p, all_out_p, all_d_p = [], [], [], []
    num_classes = 0
    
    for seed in SEEDS:
        try:
            s_r, in_r, out_r, d_r, s_p, in_p, out_p, d_p, n_c = run_single_dataset_single_seed(dataset_name, shot, seed, gpu_id, params)
            all_sim_r.append(s_r); all_in_r.append(in_r); all_out_r.append(out_r); all_d_r.append(d_r)
            all_sim_p.append(s_p); all_in_p.append(in_p); all_out_p.append(out_p); all_d_p.append(d_p)
            num_classes = n_c
        except Exception as e:
            pass

    return {
        "num_classes": num_classes,
        "sim_raw": np.mean(all_sim_r, axis=0),
        "intra_raw": np.mean(all_in_r),
        "inter_raw": np.mean(all_out_r),
        "delta_raw": np.mean(all_d_r),
        "sim_prompted": np.mean(all_sim_p, axis=0),
        "intra_prompted": np.mean(all_in_p),
        "inter_prompted": np.mean(all_out_p),
        "delta_prompted": np.mean(all_d_p)
    }

def plot_elegant_heatmap_grid(results_dict):
    print("\n🎨 开始渲染 WebKB 三剑客顶会热力图...")
    datasets = list(results_dict.keys())
    num_rows = len(datasets)
    fig, axes = plt.subplots(num_rows, 2, figsize=(11, 4.5 * num_rows), gridspec_kw={'hspace': 0.3})
    cmap = "RdBu_r" 

    for i, ds in enumerate(datasets):
        res = results_dict[ds]
        num_c = res["num_classes"]
        class_labels = [f"C{j}" for j in range(num_c)]
        
        vmin = min(res["sim_raw"].min(), res["sim_prompted"].min())
        vmax = max(res["sim_raw"].max(), res["sim_prompted"].max())

        # 左图：Before
        ax_left = axes[i, 0] if num_rows > 1 else axes[0]
        sns.heatmap(res["sim_raw"], ax=ax_left, cmap=cmap, vmin=vmin, vmax=vmax, 
                    annot=False, square=True, cbar=True, cbar_kws={"shrink": 0.8},
                    linewidths=0.5, linecolor='white', xticklabels=class_labels, yticklabels=class_labels)
        ax_left.set_ylabel(f"**{ds.capitalize()}**\nClass ID")
        if i == 0: ax_left.set_title("Before SA-OT (Averaged)", fontweight='bold', pad=15)
        
        text_raw = f"Intra: {res['intra_raw']:.2f}\nInter: {res['inter_raw']:.2f}\n$\Delta$: {res['delta_raw']:.2f}"
        ax_left.text(num_c - 0.2, num_c - 0.2, text_raw, color='black', fontsize=11,
                     ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # 右图：After
        ax_right = axes[i, 1] if num_rows > 1 else axes[1]
        sns.heatmap(res["sim_prompted"], ax=ax_right, cmap=cmap, vmin=vmin, vmax=vmax, 
                    annot=False, square=True, cbar=True, cbar_kws={"shrink": 0.8},
                    linewidths=0.5, linecolor='white', xticklabels=class_labels, yticklabels=class_labels)
        if i == 0: ax_right.set_title("After SA-OT (Averaged)", fontweight='bold', pad=15)
        
        # 加粗显示大幅提升的 Margin (Delta)
        text_prompted = f"Intra: {res['intra_prompted']:.2f}\nInter: {res['inter_prompted']:.2f}\n$\mathbf{{\Delta}}$: **{res['delta_prompted']:.2f}**"
        ax_right.text(num_c - 0.2, num_c - 0.2, text_prompted, color='black', fontsize=11, fontweight='bold',
                      ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    plt.savefig("Exp0_WebKB_Purification.pdf", format='pdf', dpi=300)
    print("🎉 WebKB 热力图绘制完成！已保存为 Exp0_WebKB_Purification.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--shot', type=int, default=1) 
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    DATASETS = ["cornell", "texas", "wisconsin"]
    
    print("="*60)
    print(f"🚀 启动 WebKB 三剑客模式: 纯 Cosine 逻辑闭环 ({args.shot}-shot)")
    print("="*60)
    
    results_dict = {}
    for ds in DATASETS:
        results_dict[ds] = run_dataset_with_seeds(ds, args.shot, args.gpu)
            
    if results_dict:
        plot_elegant_heatmap_grid(results_dict)

if __name__ == "__main__":
    main()