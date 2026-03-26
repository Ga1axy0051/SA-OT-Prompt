import torch
import torch.nn as nn

class AllInOne_Prompt(nn.Module):
    def __init__(self, in_dim, prompt_num=10):
        super().__init__()
        # All-in-one 的灵魂：虚拟的 Prompt 节点特征
        self.prompt_num = prompt_num
        self.prompt_feats = nn.Parameter(torch.empty(prompt_num, in_dim))
        nn.init.xavier_uniform_(self.prompt_feats)

    def forward(self, x, edge_index, edge_weight):
        device = x.device
        num_orig_nodes = x.size(0)
        
        # 1. 将 Prompt 节点特征拼接到原图节点特征的末尾
        new_x = torch.cat([x, self.prompt_feats], dim=0) # [N + prompt_num, in_dim]
        
        # 2. 构建 Prompt 节点与原图节点的连边 (这里采用全连接策略，让Prompt影响所有节点)
        prompt_indices = torch.arange(num_orig_nodes, num_orig_nodes + self.prompt_num, device=device)
        orig_indices = torch.arange(num_orig_nodes, device=device)
        
        # 生成 (prompt, orig) 和 (orig, prompt) 的双向连边
        grid_x, grid_y = torch.meshgrid(prompt_indices, orig_indices, indexing='ij')
        new_edges = torch.stack([grid_x.flatten(), grid_y.flatten()])
        new_edges_rev = torch.stack([grid_y.flatten(), grid_x.flatten()])
        
        # 拼接边索引和边权重 (新增加的边权重默认设为 1.0，或者也可以学习)
        new_edge_index = torch.cat([edge_index, new_edges, new_edges_rev], dim=1)
        new_edge_weight = torch.cat([edge_weight, torch.ones(new_edges.size(1) * 2, device=device)])
        
        return new_x, new_edge_index, new_edge_weight