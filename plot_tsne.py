import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np

# 学术配色
palette = sns.color_palette("Set2", 8)
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 14})

def get_tsne_plot(gpu_id, dataset_name='texas'):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"=== 正在生成 NeurIPS 正文版 2x3 阵列 ===")
    
    from utils.data_loader import load_dataset, generate_few_shot_splits
    from models.graphmae import build_model
    from models.ot_prompt import SAOTPrompt
    from models.Base import LogReg
    from torch_geometric.utils import add_self_loops
    from utils.legacy_utils import normalize_edge

    # 1. 5-shot 数据划分
    data, input_dim, output_dim = load_dataset(dataset_name, data_dir="./data/raw")
    data = data.to(device)
    data = generate_few_shot_splits(data, output_dim, shot=5, seed=42) 
    
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32, device=device)
    edge_index, edge_weight = add_self_loops(data.edge_index, edge_weight)
    edge_weight = normalize_edge(edge_index, edge_weight, data.num_nodes).to(device)
    edge_index = edge_index.to(device)

    # 2. 加载冻结底座
    base_encoder = build_model(num_hidden=256, num_features=input_dim).to(device)
    model_save_path = f'./pretrain_model/GraphMAE/{dataset_name}_hid256.pkl'
    base_encoder.load_state_dict(torch.load(model_save_path, map_location=device))
    base_encoder.eval()
    
    with torch.no_grad():
        Z_raw = base_encoder.embed(data.x, edge_index, edge_weight)

    # 3. 极速微调 SA-OT
    prompt = SAOTPrompt(data.x, input_dim, num_prompts=48, ot_epsilon=0.1, k=48).to(device)
    classifier = LogReg(256, output_dim).to(device)
    optimizer = torch.optim.Adam(list(prompt.parameters()) + list(classifier.parameters()), lr=0.005, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    prompt.train()
    classifier.train()
    for epoch in range(300):
        optimizer.zero_grad()
        x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
        c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, tau=0.0) 
        Z_prompted = base_encoder.embed(x_ad, c_idx, c_w)
        logits = classifier(Z_prompted)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    prompt.eval()
    base_encoder.eval()
    with torch.no_grad():
        x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
        c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, tau=0.0)
        Z_prompted = base_encoder.embed(x_ad, c_idx, c_w)

    # 4. t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=12, early_exaggeration=20.0, init='pca', learning_rate='auto')
    Z_raw_2d = tsne.fit_transform(Z_raw.cpu().numpy())
    Z_prompted_2d = tsne.fit_transform(Z_prompted.cpu().numpy())
    
    labels = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()

    # ---------------------------------------------------------
    # 🟢 精华提取：只画视觉收缩最强烈的 3 个类 (适配 NeurIPS 单栏)
    selected_classes = [0, 2, 3] 
    # ---------------------------------------------------------
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)) # 调整比例，让单图更大
    
    for i, c in enumerate(selected_classes):
        # ---------------- 上排：Before ----------------
        ax_b = axes[0, i]
        ax_b.scatter(Z_raw_2d[:, 0], Z_raw_2d[:, 1], color='gray', s=40, alpha=0.15, edgecolors='none')
        
        mask_c = (labels == c)
        train_mask_c = train_mask & mask_c
        unlabeled_mask_c = (~train_mask) & mask_c
        
        sns.kdeplot(x=Z_raw_2d[mask_c, 0], y=Z_raw_2d[mask_c, 1], ax=ax_b, fill=True, alpha=0.25, color=palette[c], bw_adjust=1.5, legend=False)
        ax_b.scatter(Z_raw_2d[unlabeled_mask_c, 0], Z_raw_2d[unlabeled_mask_c, 1], color=palette[c], s=80, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax_b.scatter(Z_raw_2d[train_mask_c, 0], Z_raw_2d[train_mask_c, 1], color=palette[c], marker='*', s=400, edgecolors='black', linewidths=1.2, zorder=10)
        
        ax_b.set_title(f"Class {c} (Before)", fontsize=18, fontweight='bold')
        ax_b.axis('off')

        # ---------------- 下排：After ----------------
        ax_a = axes[1, i]
        ax_a.scatter(Z_prompted_2d[:, 0], Z_prompted_2d[:, 1], color='gray', s=40, alpha=0.15, edgecolors='none')
        
        sns.kdeplot(x=Z_prompted_2d[mask_c, 0], y=Z_prompted_2d[mask_c, 1], ax=ax_a, fill=True, alpha=0.35, color=palette[c], bw_adjust=1.5, legend=False)
        ax_a.scatter(Z_prompted_2d[unlabeled_mask_c, 0], Z_prompted_2d[unlabeled_mask_c, 1], color=palette[c], s=80, alpha=0.8, edgecolors='white', linewidths=0.5)
        ax_a.scatter(Z_prompted_2d[train_mask_c, 0], Z_prompted_2d[train_mask_c, 1], color=palette[c], marker='*', s=400, edgecolors='black', linewidths=1.2, zorder=10)
        
        ax_a.set_title(f"Class {c} (After SA-OT)", fontsize=18, fontweight='bold')
        ax_a.axis('off')

    plt.tight_layout()
    plt.savefig("Exp3_tSNE_NeurIPS_Main.pdf", format='pdf', dpi=300)
    print("🎉 【正文版 2x3】绘制完成！已保存为 Exp3_tSNE_NeurIPS_Main.pdf。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='Specify GPU ID')
    args = parser.parse_args()
    get_tsne_plot(gpu_id=args.gpu)