import torch
import torch.nn as nn

class EdgePrompt(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # 学习一个边打分器，输入为相连两个节点的特征拼接
        self.edge_scorer = nn.Linear(in_dim * 2, 1)
        
    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index
        edge_feats = torch.cat([x[src], x[dst]], dim=-1)
        # 用 Sigmoid 将打分压缩到 0~1，作为边的保留/增强系数
        prompt_scores = torch.sigmoid(self.edge_scorer(edge_feats)).squeeze(-1)
        return edge_weight * prompt_scores

class EdgePrompt_plus(nn.Module):
    def __init__(self, in_dim, num_bases=5):
        super().__init__()
        # EdgePrompt+ 维护一组边特征的“基向量” (Basis Vectors)
        self.bases = nn.Parameter(torch.empty(num_bases, in_dim * 2))
        nn.init.xavier_uniform_(self.bases)
        self.attn = nn.Linear(in_dim * 2, num_bases)
        self.scorer = nn.Linear(in_dim * 2, 1)

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index
        edge_feats = torch.cat([x[src], x[dst]], dim=-1) # [E, 2d]
        
        # 计算注意力权重并融合基向量
        attn_weights = torch.softmax(self.attn(edge_feats), dim=-1) # [E, num_bases]
        prompt_edge_feats = torch.matmul(attn_weights, self.bases) # [E, 2d]
        
        # 结合原始边特征并打分
        combined_feats = edge_feats + prompt_edge_feats
        prompt_scores = torch.sigmoid(self.scorer(combined_feats)).squeeze(-1)
        
        return edge_weight * prompt_scores