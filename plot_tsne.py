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
    
    print(f"=== 正在运行实验三：t-SNE Manifold Repair on {dataset_name} (GPU: {gpu_id}) ===")
    
    from utils.data_loader import load_dataset, generate_few_shot_splits
    from models.graphmae import build_model
    from models.ot_prompt import SAOTPrompt
    from models.Base import LogReg
    from torch_geometric.utils import add_self_loops
    from utils.legacy_utils import normalize_edge

    # 1. 加载数据并生成 1-shot 划分
    data, input_dim, output_dim = load_dataset(dataset_name, data_dir="./data/raw")
    data = data.to(device)
    data = generate_few_shot_splits(data, output_dim, shot=1, seed=42) # 1-shot，每类1个点
    
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

    # 3. 极速微调 SA-OT (使用打擂台搜出的最优超参)
    prompt = SAOTPrompt(data.x, input_dim, num_prompts=35, ot_epsilon=0.1, k=35).to(device)
    classifier = LogReg(256, output_dim).to(device)
    # Texas 1-shot 最优学习率 0.009, weight decay 0.001
    optimizer = torch.optim.Adam(list(prompt.parameters()) + list(classifier.parameters()), lr=0.009, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    print("正在使用最优超参深度修复流形空间...")
    prompt.train()
    classifier.train()
    for epoch in range(250): # 增加 epoch 确保流形彻底拉开
        optimizer.zero_grad()
        x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
        # Texas 异配图最优 tau 为 0.0，彻底斩断毒性边
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

    # 4. 严谨的 t-SNE 降维
    print("正在进行 t-SNE 降维...")
    # 使用 pca 初始化让降维更稳定，更能反映真实拓扑
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    
    Z_raw_2d = tsne.fit_transform(Z_raw.cpu().numpy())
    Z_prompted_2d = tsne.fit_transform(Z_prompted.cpu().numpy())
    
    labels = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()

    # 5. 绘图 (区分无标签节点和 1-shot 有标签锚点)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ---------------- 左图：Before ----------------
    # 画普通的无标签节点
    axes[0].scatter(Z_raw_2d[~train_mask, 0], Z_raw_2d[~train_mask, 1], 
                    c=[palette[l] for l in labels[~train_mask]], s=50, alpha=0.5, edgecolors='none')
    # 画 1-shot 真实标注节点（大星星）
    axes[0].scatter(Z_raw_2d[train_mask, 0], Z_raw_2d[train_mask, 1], 
                    c=[palette[l] for l in labels[train_mask]], marker='*', s=500, edgecolors='black', linewidths=1.5)
    axes[0].set_title("Before SA-OT: Entangled Manifold", fontsize=16)
    axes[0].axis('off')
    
    # ---------------- 右图：After ----------------
    # 画普通的无标签节点
    axes[1].scatter(Z_prompted_2d[~train_mask, 0], Z_prompted_2d[~train_mask, 1], 
                    c=[palette[l] for l in labels[~train_mask]], s=50, alpha=0.5, edgecolors='none')
    # 画 1-shot 真实标注节点（大星星）
    axes[1].scatter(Z_prompted_2d[train_mask, 0], Z_prompted_2d[train_mask, 1], 
                    c=[palette[l] for l in labels[train_mask]], marker='*', s=500, edgecolors='black', linewidths=1.5, label='1-Shot Labeled Anchors')
    axes[1].set_title("After SA-OT: Semantic Disentanglement", fontsize=16)
    axes[1].axis('off')
    
    # 添加图例
    axes[1].legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Exp3_tSNE_Manifold.pdf", format='pdf', dpi=300)
    print("🎉 图 3 绘制完成！已保存为 Exp3_tSNE_Manifold.pdf。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='Specify GPU ID')
    args = parser.parse_args()
    get_tsne_plot(gpu_id=args.gpu)