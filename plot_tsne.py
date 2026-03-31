import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# 学术配色
palette = sns.color_palette("Set2", 8)
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})

def get_tsne_plot(gpu_id, dataset_name='texas'):
    # 强行屏蔽其他显卡，PyTorch 眼里就只有这一张卡了
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"=== 正在运行实验三：t-SNE Manifold Repair on {dataset_name} (GPU: {gpu_id}) ===")
    
    # 延迟导入，防止在设 CUDA_VISIBLE_DEVICES 之前初始化 CUDA
    from utils.data_loader import load_dataset, generate_few_shot_splits  # 🟢 引入 few-shot 划分
    from models.graphmae import build_model
    from models.ot_prompt import SAOTPrompt
    from models.Base import LogReg
    from torch_geometric.utils import add_self_loops
    from utils.legacy_utils import normalize_edge

    # 1. 加载基础数据
    data, input_dim, output_dim = load_dataset(dataset_name, data_dir="./data/raw")
    data = data.to(device)
    
    # 🟢 极其关键的一步：生成 1D 的 1-shot 掩码！解决维度越界报错！
    data = generate_few_shot_splits(data, output_dim, shot=1, seed=42)
    
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

    # 3. 快速微调 SA-OT
    prompt = SAOTPrompt(data.x, input_dim, num_prompts=10, ot_epsilon=0.1, k=50).to(device)
    classifier = LogReg(256, output_dim).to(device)
    optimizer = torch.optim.Adam(list(prompt.parameters()) + list(classifier.parameters()), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    
    print("快速微调 SA-OT...")
    prompt.train()
    classifier.train()
    for _ in range(100):
        optimizer.zero_grad()
        x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
        c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, tau=0.1)
        Z_prompted = base_encoder.embed(x_ad, c_idx, c_w)
        logits = classifier(Z_prompted)
        # 🟢 现在 data.train_mask 是一维布尔值，可以完美切片了！
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    prompt.eval()
    classifier.eval()
    with torch.no_grad():
        x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
        c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, tau=0.1)
        Z_prompted = base_encoder.embed(x_ad, c_idx, c_w)
        prompt_centers = classifier.fc.weight.detach()

    # 4. t-SNE 降维
    print("正在进行 t-SNE 降维 (这需要十几秒)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    Z_raw_2d = tsne.fit_transform(Z_raw.cpu().numpy())
    
    combined_Z = torch.cat([Z_prompted, prompt_centers], dim=0).cpu().numpy()
    combined_2d = tsne.fit_transform(combined_Z)
    Z_prompted_2d = combined_2d[:data.num_nodes]
    centers_2d = combined_2d[data.num_nodes:]
    
    labels = data.y.cpu().numpy()

    # 5. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=Z_raw_2d[:, 0], y=Z_raw_2d[:, 1], hue=labels, palette=palette, s=60, alpha=0.8, ax=axes[0], legend=False)
    axes[0].set_title("Before SA-OT: Entangled Manifold", fontsize=16)
    axes[0].axis('off')
    
    sns.scatterplot(x=Z_prompted_2d[:, 0], y=Z_prompted_2d[:, 1], hue=labels, palette=palette, s=60, alpha=0.8, ax=axes[1], legend=False)
    axes[1].scatter(centers_2d[:, 0], centers_2d[:, 1], c='gold', marker='*', s=600, edgecolor='black', linewidths=1.5, label='Prompt Anchors', zorder=5)
    axes[1].set_title("After SA-OT: Semantic Disentanglement", fontsize=16)
    axes[1].axis('off')
    axes[1].legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Exp3_tSNE_Manifold.pdf", format='pdf', dpi=300)
    print("🎉 图 3 绘制完成！已保存为 Exp3_tSNE_Manifold.pdf。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='Specify GPU ID (e.g., 2 or 5)')
    args = parser.parse_args()
    get_tsne_plot(gpu_id=args.gpu)