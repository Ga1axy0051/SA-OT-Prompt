import torch
import torch.nn as nn
from torch_geometric.utils import coalesce

class HSGPPT_Prompt(nn.Module):
    def __init__(self, in_dim, num_nodes_prompt=10, tau_inner=0.2, tau_cross=0.4, device='cuda'):
        """
        HS-GPPT (Aligning the Spectrum) Prompt 核心机制复刻
        根据论文 Section 4.3: Spectral-Aligned Prompt Tuning 实现
        """
        super(HSGPPT_Prompt, self).__init__()
        self.num_nodes_prompt = num_nodes_prompt
        self.tau_inner = tau_inner
        self.tau_cross = tau_cross
        self.device = device
        
        # 定义可学习的 Prompt 节点特征
        self.prompt_features = nn.Parameter(torch.empty(num_nodes_prompt, in_dim).to(device))
        nn.init.xavier_uniform_(self.prompt_features)
        
    def forward(self, x, edge_index):
        num_orig_nodes = x.size(0)
        
        # ---------------------------------------------------------
        # 1. 论文 Eq.8: 节点特征分布归一化 (Prompt Feature Normalization)
        # ---------------------------------------------------------
        mu_p = self.prompt_features.mean(dim=0, keepdim=True)
        sigma_p = self.prompt_features.std(dim=0, keepdim=True) + 1e-8
        
        mu_o = x.mean(dim=0, keepdim=True)
        sigma_o = x.std(dim=0, keepdim=True) + 1e-8
        
        # 强制对齐分布
        p_prime = (self.prompt_features - mu_p) / sigma_p * sigma_o + mu_o
        
        # 将 Prompt 节点追加到全图最末尾
        x_combined = torch.cat([x, p_prime], dim=0)
        
        # ---------------------------------------------------------
        # 2. 构造 Prompt 内部连边 (Inner Edges)
        # ---------------------------------------------------------
        sim_inner = torch.sigmoid(torch.mm(p_prime, p_prime.t()))
        adj_inner = (sim_inner > self.tau_inner).float()
        edge_index_inner = adj_inner.nonzero().t()
        # 加上索引偏移量
        edge_index_inner += num_orig_nodes 
        
        # ---------------------------------------------------------
        # 3. 构造跨图连边 (Cross Edges) - 最核心的拓扑重构点
        # ---------------------------------------------------------
        sim_cross = torch.sigmoid(torch.mm(p_prime, x.t())) # shape: [num_prompt, num_orig]
        adj_cross = (sim_cross > self.tau_cross).float()
        cross_edges = adj_cross.nonzero() # shape: [num_edges, 2]
        
        # cross_edges[:, 0] 是 prompt 节点索引，cross_edges[:, 1] 是原始节点索引
        src_cross = cross_edges[:, 0] + num_orig_nodes
        dst_cross = cross_edges[:, 1]
        
        # 构造无向交叉边
        edge_index_cross_1 = torch.stack([src_cross, dst_cross], dim=0)
        edge_index_cross_2 = torch.stack([dst_cross, src_cross], dim=0) 
        
        # ---------------------------------------------------------
        # 4. 图拓扑大一统
        # ---------------------------------------------------------
        final_edge_index = torch.cat([edge_index, edge_index_inner, edge_index_cross_1, edge_index_cross_2], dim=1)
        final_edge_index = coalesce(final_edge_index, num_nodes=num_orig_nodes + self.num_nodes_prompt)
        
        final_edge_weight = torch.ones(final_edge_index.size(1), device=self.device)
        
        return x_combined, final_edge_index, final_edge_weight