import torch
import torch.nn as nn

class ProNoG_Prompt(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, device='cuda'):
        """
        ProNoG (Non-Homophilic Graph Pre-Training and Prompt Learning) 核心机制复刻
        依赖条件网络 (Condition Network) 为每个节点生成专属的特征级 Prompt 向量
        """
        super(ProNoG_Prompt, self).__init__()
        self.device = device
        
        # 1. 节点级别的特征映射 (用于计算子图 Readout 的相似度权重)
        self.sim_proj = nn.Linear(in_dim, hidden_dim)
        self.sim_vector = nn.Linear(hidden_dim, 1)
        
        # 2. 条件网络 (Condition Net): 将 Readout 映射为专属 Prompt
        self.condition_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x, edge_index):
        # ---------------------------------------------------------
        # 步骤 1: 邻域子图 Readout (加权聚合)
        # ---------------------------------------------------------
        # 计算中心节点和邻居的相似度权重
        h = torch.tanh(self.sim_proj(x))
        weights = torch.sigmoid(self.sim_vector(h).squeeze()) # shape: [N]
        
        row, col = edge_index
        edge_weights = weights[col]
        
        # 将邻居特征加权聚合到中心节点 (模拟 ProNoG 的 Subgraph Readout)
        readout = torch.zeros_like(x)
        readout.scatter_add_(0, row.unsqueeze(-1).expand(-1, x.size(1)), x[col] * edge_weights.unsqueeze(-1))
        
        # ---------------------------------------------------------
        # 步骤 2: 条件网络生成专属 Prompt
        # ---------------------------------------------------------
        node_specific_prompts = self.condition_net(readout) # shape: [N, in_dim]
        
        # ---------------------------------------------------------
        # 步骤 3: 特征级 Prompt 融合 (Feature-level Injection)
        # ProNoG 不修改拓扑结构，只修改节点特征
        # ---------------------------------------------------------
        x_prompted = x + node_specific_prompts
        
        # 拓扑结构原封不动返回
        return x_prompted, edge_index, None