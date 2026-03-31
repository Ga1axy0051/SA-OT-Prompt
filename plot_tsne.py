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
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})

def get_tsne_plot(gpu_id, dataset_name='texas'):
    # 强行屏蔽其他显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"=== 正在运行进阶版实验三：5-Shot t-SNE Manifold Repair on {dataset_name} ===")
    
    from utils.data_loader import load_dataset, generate_few_shot_splits
    from models.graphmae import build_model
    from models.ot_prompt import SAOTPrompt
    from models.Base import LogReg
    from torch_geometric.utils import add_self_loops
    from utils.legacy_utils import normalize_edge

    # 1. 🟢 升级为 5-shot，提供足够的梯度拉扯力
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

    # 3. 极速微调 SA-OT (Texas 5-shot 最优超参: lr=0.005, wd=0.001, tau=0.0, k=48)
    prompt = SAOTPrompt(data.x, input_dim, num_prompts=48, ot_epsilon=0.1, k=48).to(device)
    classifier = LogReg(256, output_dim).to(device)
    optimizer = torch.optim.Adam(list(prompt.parameters()) + list(classifier.parameters()), lr=0.005, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    print("正在强力重塑流形空间...")
    prompt.train()
    classifier.train()
    for epoch in range(300): # 300 epoch 确保充分聚类
        optimizer.zero_grad()
        x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
        c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, tau=0.0) 
        Z_prompted = base_encoder.embed(x_ad, c_idx, c_w)
        logits = classifier(Z_prompted)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # 提取修复后的表征
    prompt.eval()
    base_encoder.eval()
    with torch.no_grad():
        x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
        c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, tau=0.0)
        Z_prompted = base_encoder.embed(x_ad, c_idx, c_w)

    # 4. 🟢 严谨的 t-SNE 降维 (特调小图专用参数)
    print("正在进行高精度 t-SNE 降维...")
    tsne = TSNE(n_components=2, random_state=42, 
                perplexity=12,               # 专为 183 节点定制的小困惑度，强化局部抱团
                early_exaggeration=20.0,     # 早期夸大倍数，强制推开不同类
                init='pca', learning_rate='auto')
    
    Z_raw_2d = tsne.fit_transform(Z_raw.cpu().numpy())
    Z_prompted_2d = tsne.fit_transform(Z_prompted.cpu().numpy())
    
    labels = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()

    # 5. 🟢 影院级绘图增强
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：Before (混沌状态，使用白边增强质感)
    axes[0].scatter(Z_raw_2d[~train_mask, 0], Z_raw_2d[~train_mask, 1], 
                    c=[palette[l] for l in labels[~train_mask]], s=70, alpha=0.6, edgecolors='white', linewidths=0.5)
    axes[0].scatter(Z_raw_2d[train_mask, 0], Z_raw_2d[train_mask, 1], 
                    c=[palette[l] for l in labels[train_mask]], marker='*', s=400, edgecolors='black', linewidths=1.0)
    axes[0].set_title("Before SA-OT: Entangled Manifold", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # 右图：After (解耦状态)
    axes[1].scatter(Z_prompted_2d[~train_mask, 0], Z_prompted_2d[~train_mask, 1], 
                    c=[palette[l] for l in labels[~train_mask]], s=70, alpha=0.8, edgecolors='white', linewidths=0.5)
    axes[1].scatter(Z_prompted_2d[train_mask, 0], Z_prompted_2d[train_mask, 1], 
                    c=[palette[l] for l in labels[train_mask]], marker='*', s=400, edgecolors='black', linewidths=1.2, label='5-Shot Labeled Anchors')
    axes[1].set_title("After SA-OT: Semantic Disentanglement", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    axes[1].legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Exp3_tSNE_Manifold.pdf", format='pdf', dpi=300)
    print("🎉 图 3 绘制完成！已保存为 Exp3_tSNE_Manifold.pdf。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='Specify GPU ID')
    args = parser.parse_args()
    get_tsne_plot(gpu_id=args.gpu)