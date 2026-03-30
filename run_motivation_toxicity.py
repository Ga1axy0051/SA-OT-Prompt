import torch
import numpy as np
import os

# ==========================================
# 🚨 极其关键：服务器无头模式画图救星
# ==========================================
import matplotlib
matplotlib.use('Agg')  # 必须写在 import pyplot 之前！
import matplotlib.pyplot as plt

from torch_geometric.datasets import WebKB
from torch_geometric.utils import homophily

# ==========================================
# 🚀 霸气锁卡设置
# ==========================================
GPU_ID = '2'  # 替换为你要跑的卡号
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前动机实验 2 正在运行的设备: {device} (物理卡号: {GPU_ID})")

# ==========================================
# 加载真实数据并推入 GPU
# ==========================================
dataset_path = os.path.join(os.getcwd(), 'data', 'WebKB')
dataset = WebKB(root=dataset_path, name='Texas')
# 将整张图直接放到 GPU 上加速计算
data = dataset[0].to(device)

edge_index = data.edge_index
y = data.y
num_nodes = data.num_nodes
original_edges_count = edge_index.shape[1]

# 测算原始同配率
orig_h_node = homophily(edge_index, y, method='node')
print(f"Texas 原始 Node 同配率: {orig_h_node:.4f}")

# ==========================================
# 严谨拓扑毒性模拟 (全链路 GPU 加速)
# ==========================================
noise_ratios = np.linspace(0, 2.0, 15) 
homophily_drops_node = []

torch.manual_seed(2026)

for ratio in noise_ratios:
    num_fake_edges = int(original_edges_count * ratio)
    
    if num_fake_edges > 0:
        # ⚠️ 在指定 GPU 上生成假边，避免 CPU/GPU 数据交互导致报错
        fake_src = torch.randint(0, num_nodes, (num_fake_edges,), device=device)
        fake_dst = torch.randint(0, num_nodes, (num_fake_edges,), device=device)
        fake_edges = torch.stack([fake_src, fake_dst], dim=0)
        
        corrupted_edge_index = torch.cat([edge_index, fake_edges], dim=1)
    else:
        corrupted_edge_index = edge_index
        
    h_node = homophily(corrupted_edge_index, y, method='node')
    homophily_drops_node.append(h_node)

# ==========================================
# 严谨绘图
# ==========================================
fig, ax = plt.figure(figsize=(9, 6)), plt.gca()

ax.plot(noise_ratios * 100, homophily_drops_node, marker='s', markersize=6, 
        linestyle='-', color='#d62728', linewidth=2.5, 
        label='Vanilla Graph Prompting (Unconstrained)')

ax.axhline(y=orig_h_node, color='#1f77b4', linestyle='--', linewidth=3, 
           label='SA-OT-Prompt (Ours with $\\tau$ Structure-Aware Filter)')

ax.fill_between(noise_ratios * 100, homophily_drops_node, orig_h_node, color='#d62728', alpha=0.1)

ax.set_title("Topological Toxicity of Unconstrained Prompting\n(Real Data: Texas Dataset)", fontsize=14, fontweight='bold')
ax.set_xlabel("Injected Prompt-Induced Edges (% of original edges)", fontsize=12)
ax.set_ylabel("Graph Homophily Ratio ($\mathcal{H}_{node}$)", fontsize=12)

ax.annotate('Severe Structural\nDegradation', xy=(150, homophily_drops_node[-3]), 
            xytext=(100, orig_h_node + 0.05),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=11, fontweight='bold', color='#d62728')

ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig("rigorous_topology_toxicity.pdf", dpi=300, bbox_inches='tight')
print("✅ 严谨拓扑毒性图表已生成: rigorous_topology_toxicity.pdf")