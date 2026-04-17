import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce

class DAPrompt_Prompt(nn.Module):
    def __init__(self, in_dim, num_classes, num_structs=2, outer_thre=0.2, device='cuda'):
        """
        DAPrompt 的全图适配版 (Full-Graph Adapted)
        """
        super(DAPrompt_Prompt, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes # 语义 Token 数量 (等于类别数)
        self.num_structs = num_structs # 结构 Token 数量
        self.outer_thre = outer_thre   # 余弦相似度截断阈值
        self.device = device
        
        # 结构 Prompt (用于重构拓扑)
        self.structure_prompt = nn.Parameter(torch.empty(self.num_structs, self.in_dim).to(device))
        nn.init.xavier_uniform_(self.structure_prompt)
        
        # 语义 Prompt (用于语义对齐与锚点连接)
        self.semantics_prompt = nn.Parameter(torch.empty(self.num_classes, self.in_dim).to(device))
        nn.init.xavier_uniform_(self.semantics_prompt)

    def inner_edge(self, X, thre):
        """计算余弦相似度并根据阈值截断生成连边"""
        X_norm = F.normalize(X, p=2, dim=1)
        sim = torch.mm(X_norm, X_norm.t())
        sim = torch.sigmoid(sim) # 遵循 DAPrompt 源码，经过 Sigmoid
        adj = (sim > thre).float()
        return adj.nonzero().t()

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # ---------------------------------------------------------
        # 1. 结构重构 (Structure Prompting)
        # 将结构 Prompt 追加到特征末尾，保证原节点索引 0~N-1 不变
        # ---------------------------------------------------------
        x_struct = torch.cat([x, self.structure_prompt], dim=0)
        num_nodes_struct = num_nodes + self.num_structs
        
        # DAPrompt 核心：通过无约束的余弦相似度计算全图新边
        sim_edge_index = self.inner_edge(x_struct, self.outer_thre)
        
        # 融合原始边与基于相似度的新边
        merged_edge_index = torch.cat([edge_index, sim_edge_index], dim=1)
        merged_edge_index = coalesce(merged_edge_index, num_nodes=num_nodes_struct)
        
        # ---------------------------------------------------------
        # 2. 语义对齐 (Semantics Prompting)
        # 将语义 Prompt 继续追加到最末尾
        # ---------------------------------------------------------
        x_final = torch.cat([x_struct, self.semantics_prompt], dim=0)
        num_nodes_final = num_nodes_struct + self.num_classes
        
        # 计算每个原始节点与所有语义 Token 的相似度
        x_norm = F.normalize(x, p=2, dim=1)
        s_norm = F.normalize(self.semantics_prompt, p=2, dim=1)
        cos_sim = torch.mm(x_norm, s_norm.t()) # shape: [num_nodes, num_classes]
        
        # 遵循 DAPrompt 源码的 argmax 逻辑：每个节点连向得分最高的语义 Token
        sim = torch.sigmoid(cos_sim)
        best_class = sim.argmax(dim=1) # shape: [num_nodes]
        
        # 生成锚点连边
        target_nodes = torch.arange(num_nodes, device=self.device)
        semantics_nodes = num_nodes_struct + best_class
        
        target_edges = torch.stack([target_nodes, semantics_nodes], dim=0)
        target_edges = torch.cat([target_edges, target_edges.flip(0)], dim=1) # 无向化
        
        # 最终图拓扑：合并所有边
        final_edge_index = torch.cat([merged_edge_index, target_edges], dim=1)
        final_edge_index = coalesce(final_edge_index, num_nodes=num_nodes_final)
        
        # 给所有新生成的边赋予权重 1.0 (为兼容基座)
        final_edge_weight = torch.ones(final_edge_index.size(1), device=self.device)
        
        return x_final, final_edge_index, final_edge_weight