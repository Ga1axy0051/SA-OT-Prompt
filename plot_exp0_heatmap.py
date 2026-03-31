import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

# 🟢 顶会级绘图全局设置
sns.set_theme(style="white") 
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14, 
    'pdf.fonttype': 42, 
    'ps.fonttype': 42,
})

def get_class_cosine_similarity(Z, labels, num_classes):
    """计算类别间的平均余弦相似度矩阵"""
    # 1. 特征 L2 归一化 (极其关键，消除欧式距离灾难)
    Z_norm = F.normalize(Z, p=2, dim=1)
    
    sim_matrix = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(num_classes):
            # 获取属于类 i 和类 j 的所有节点特征
            Z_i = Z_norm[labels == i]
            Z_j = Z_norm[labels == j]
            
            # 如果某个类没有节点，跳过
            if len(Z_i) == 0 or len(Z_j) == 0:
                continue
                
            # 计算全连接的余弦相似度矩阵: (N_i, D) x (D, N_j) -> (N_i, N_j)
            cos_sim = torch.mm(Z_i, Z_j.t())
            
            # 取平均值作为类 i 和类 j 的整体相似度
            sim_matrix[i, j] = cos_sim.mean().item()
            
    return sim_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='Specify GPU ID')
    args = parser.parse_args()
    
    dataset_name = 'texas'
    shot = 5
    seed = 42
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    from utils.data_loader import load_dataset, generate_few_shot_splits
    from models.graphmae import build_model
    from models.ot_prompt import SAOTPrompt
    from models.Base import LogReg
    from torch_geometric.utils import add_self_loops
    from utils.legacy_utils import normalize_edge

    print(f"🚀 开始采集并绘制【实验零：特征净化热力图】 (Dataset: {dataset_name.upper()})")

    # 1. 准备数据
    data, input_dim, output_dim = load_dataset(dataset_name, data_dir="./data/raw")
    data = data.to(device)
    data = generate_few_shot_splits(data, output_dim, shot=shot, seed=seed) 
    num_classes = output_dim
    
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32, device=device)
    edge_index, edge_weight = add_self_loops(data.edge_index, edge_weight)
    edge_weight = normalize_edge(edge_index, edge_weight, data.num_nodes).to(device)
    edge_index = edge_index.to(device)

    # 2. 提取 GraphMAE (Before) 特征
    base_encoder = build_model(num_hidden=256, num_features=input_dim).to(device)
    model_save_path = f'./pretrain_model/GraphMAE/{dataset_name}_hid256.pkl'
    base_encoder.load_state_dict(torch.load(model_save_path, map_location=device))
    base_encoder.eval()
    
    with torch.no_grad():
        Z_raw = base_encoder.embed(data.x, edge_index, edge_weight)
    sim_raw = get_class_cosine_similarity(Z_raw, data.y, num_classes)

    # 3. SA-OT 微调 (After)
    prompt = SAOTPrompt(data.x, input_dim, num_prompts=48, ot_epsilon=0.1, k=48).to(device)
    classifier = LogReg(256, output_dim).to(device)
    optimizer = torch.optim.Adam(list(prompt.parameters()) + list(classifier.parameters()), lr=0.005, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    prompt.train()
    classifier.train()
    for epoch in range(200): 
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
    sim_prompted = get_class_cosine_similarity(Z_prompted, data.y, num_classes)

    # ==========================================
    # 🎨 顶会级热力图绘制 (冷暖高对比)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # 统一色标范围，确保对比公平
    vmin = min(sim_raw.min(), sim_prompted.min())
    vmax = max(sim_raw.max(), sim_prompted.max())
    
    # 使用 RdYlBu_r 经典冷暖色调 (红高蓝低)
    cmap = "RdYlBu_r" 
    
    class_labels = [f"C{i}" for i in range(num_classes)]

    # 左图：Before
    sns.heatmap(sim_raw, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax, 
                annot=False, square=True, cbar_kws={"shrink": 0.8},
                xticklabels=class_labels, yticklabels=class_labels)
    axes[0].set_title("Before SA-OT: Feature Entanglement", fontweight='bold', pad=15)
    axes[0].set_xlabel("Class ID")
    axes[0].set_ylabel("Class ID")

    # 右图：After
    sns.heatmap(sim_prompted, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax, 
                annot=False, square=True, cbar_kws={"shrink": 0.8},
                xticklabels=class_labels, yticklabels=class_labels)
    axes[1].set_title("After SA-OT: Semantic Purification", fontweight='bold', pad=15)
    axes[1].set_xlabel("Class ID")
    axes[1].set_ylabel("Class ID")

    plt.tight_layout()
    plt.savefig("Exp0_Cosine_Heatmap.pdf", format='pdf', dpi=300)
    print("🎉 图表绘制完成！已保存为 Exp0_Cosine_Heatmap.pdf")

if __name__ == "__main__":
    main()