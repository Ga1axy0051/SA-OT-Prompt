import torch
import torch.nn as nn

class AllInOne_Prompt(nn.Module):
    def __init__(self, in_dim, prompt_num=10):
        super().__init__()
        # All-in-one 的灵魂：向图中插入的虚拟 Prompt 节点特征
        self.prompt_num = prompt_num
        self.prompt_feats = nn.Parameter(torch.empty(prompt_num, in_dim))
        nn.init.xavier_uniform_(self.prompt_feats)

    def forward(self, x, edge_index, edge_weight):
        device = x.device
        num_orig_nodes = x.size(0)
        
        # 1. 将 Prompt 节点特征拼接到原图节点特征的末尾
        new_x = torch.cat([x, self.prompt_feats], dim=0) # 尺寸变为 [N + prompt_num, in_dim]
        
        # 2. 构建 Prompt 节点与原图节点的全连接边
        prompt_indices = torch.arange(num_orig_nodes, num_orig_nodes + self.prompt_num, device=device)
        orig_indices = torch.arange(num_orig_nodes, device=device)
        
        # 生成双向连边 (虚拟节点 -> 真实节点，真实节点 -> 虚拟节点)
        grid_x, grid_y = torch.meshgrid(prompt_indices, orig_indices, indexing='ij')
        new_edges = torch.stack([grid_x.flatten(), grid_y.flatten()])
        new_edges_rev = torch.stack([grid_y.flatten(), grid_x.flatten()])
        
        # 3. 拼接原来的边和新增加的边
        new_edge_index = torch.cat([edge_index, new_edges, new_edges_rev], dim=1)
        # 新增加的边权重默认为 1.0
        new_edge_weight = torch.cat([edge_weight, torch.ones(new_edges.size(1) * 2, device=device)])
        
        return new_x, new_edge_index, new_edge_weight