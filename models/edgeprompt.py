import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

class EdgePrompt(nn.Module):
    def __init__(self, in_dim):
        super(EdgePrompt, self).__init__()
        self.global_prompt = nn.Parameter(torch.empty(1, in_dim))
        # 🚨 核心修复 1：降低初始化倍率，防止初始状态直接淹没原特征
        nn.init.xavier_uniform_(self.global_prompt, gain=0.1)

    def forward(self, x, edge_index):
        src, dst = edge_index
        edge_prompt = self.global_prompt.expand(edge_index.size(1), -1)
        
        node_prompt = torch.zeros_like(x)
        node_prompt.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(1)), edge_prompt)
        
        # 🚨 核心修复 2：求均值！消除度数爆炸导致的高维溢出！
        deg = degree(dst, x.size(0), dtype=x.dtype).clamp(min=1).unsqueeze(-1)
        node_prompt = node_prompt / deg
        
        return x + node_prompt

class EdgePrompt_plus(nn.Module):
    def __init__(self, in_dim, num_anchors=5):
        super(EdgePrompt_plus, self).__init__()
        self.anchor_prompt = nn.Parameter(torch.empty(num_anchors, in_dim))
        self.w = nn.Linear(2 * in_dim, num_anchors)
        
        # 🚨 核心修复 1
        nn.init.xavier_uniform_(self.anchor_prompt, gain=0.1)
        self.w.reset_parameters()

    def forward(self, x, edge_index):
        src, dst = edge_index
        combined_x = torch.cat([x[src], x[dst]], dim=-1)
        
        b = F.softmax(F.leaky_relu(self.w(combined_x)), dim=1)
        edge_prompt = b.mm(self.anchor_prompt) 
        
        node_prompt = torch.zeros_like(x)
        node_prompt.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(1)), edge_prompt)
        
        # 🚨 核心修复 2：求均值！
        deg = degree(dst, x.size(0), dtype=x.dtype).clamp(min=1).unsqueeze(-1)
        node_prompt = node_prompt / deg
        
        return x + node_prompt