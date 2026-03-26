import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree, coalesce
from sklearn.neighbors import kneighbors_graph
from layers.sinkhorn import SinkhornOT

class SAOTPrompt(nn.Module):
    def __init__(self, x, in_channels, num_prompts, ot_epsilon=0.1, k=50):
        super(SAOTPrompt, self).__init__()
        
        self.prompt_tokens = nn.Parameter(torch.Tensor(num_prompts, in_channels))
        nn.init.xavier_uniform_(self.prompt_tokens)
        self.alpha_feat = nn.Parameter(torch.tensor(0.01)) 
        self.ot_layer = SinkhornOT(epsilon=ot_epsilon, max_iters=20)
        self.gamma = nn.Parameter(torch.tensor(0.5))

        knn_adj = kneighbors_graph(x.cpu().numpy(), k, metric='cosine')
        knn_adj = knn_adj.tocoo()
        edge_index = torch.tensor(np.vstack([knn_adj.row, knn_adj.col]), dtype=torch.long)
        self.pt_edge_index = edge_index.to(x.device)

    def forward(self, x, edge_index, edge_weight):
        # 1. 结构化特征预聚合 (GCN-style Aggregation)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        x_aggr = torch.zeros_like(x)
        x_aggr.scatter_add_(0, row.view(-1, 1).expand(-1, x.size(1)), x[col] * norm_weight.view(-1, 1))
        
        # 融合原始特征与结构特征
        x_struct = (1 - self.gamma) * x + self.gamma * x_aggr
        
        # ============================================================
        # 👑 核心改进：OT 引擎的“防爆”归一化
        # ============================================================
        # 在送入 OT 之前，对特征进行 L2 归一化。
        # 这样做是为了让代价矩阵 C 的计算基于余弦距离/归一化欧式距离，
        # 彻底解决不同维度特征量级差异导致的数值不稳定问题。
        x_struct_norm = F.normalize(x_struct, p=2, dim=1)
        # 如果 prompt_tokens 也是学习出来的，建议也对其做归一化，保证 Cost 计算在同一量级
        prompt_tokens_norm = F.normalize(self.prompt_tokens, p=2, dim=1)
        
        # 执行最优传输：基于归一化特征计算传输计划 T_star
        T_star, ot_loss = self.ot_layer(x_struct_norm, prompt_tokens_norm)
        
        # 2. 特征增强 (Feature Adaptation)
        # 利用传输计划从提示词池中提取信息
        prompt_message = x.size(0) * torch.matmul(T_star, self.prompt_tokens)
        x_adapted = x + self.alpha_feat * prompt_message
        
        # 3. 提示边权重生成 (Prompt Edge Weight Generation)
        # 使用余弦相似度作为新生成边的权重，同样基于归一化特征
        x_final_norm = F.normalize(x_adapted, p=2, dim=1)
        pt_row, pt_col = self.pt_edge_index
        
        # 计算余弦相似度：s(i,j) = <x_i, x_j> / (||x_i|| * ||x_j||)
        pt_edge_weight = torch.sum(x_final_norm[pt_row] * x_final_norm[pt_col], dim=1)
        
        # 激活与归一化：ReLU 确保权重非负，后续除以 max 确保数值范围在 [0, 1]
        pt_edge_weight = F.relu(pt_edge_weight)
        pt_edge_weight = pt_edge_weight / (pt_edge_weight.max() + 1e-8)

        return x_adapted, ot_loss, self.pt_edge_index, pt_edge_weight

    def edge_fuse(self, index_ori, weight_ori, index_pt, weight_pt, tau):
        # 👑 正确的逻辑：tau 是 Prompt 的权重！
        weight_ori = weight_ori * (1 - tau)
        weight_pt = weight_pt * tau
        
        comb_index = torch.cat([index_ori, index_pt], dim=1)
        comb_weight = torch.cat([weight_ori, weight_pt], dim=0)
        return coalesce(comb_index, comb_weight, reduce='add')