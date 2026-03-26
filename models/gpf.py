import torch
import torch.nn as nn
import torch.nn.functional as F

class GPF_Prompt(nn.Module):
    def __init__(self, in_dim):
        super(GPF_Prompt, self).__init__()
        # 论文定义：GPF 使用一个全局共享的 Prompt 向量加到所有节点特征上
        self.global_prompt = nn.Parameter(torch.empty(1, in_dim))
        nn.init.xavier_uniform_(self.global_prompt)

    def forward(self, x):
        # 直接与原始特征相加
        return x + self.global_prompt

class GPF_plus_Prompt(nn.Module):
    def __init__(self, in_dim, prompt_num=5):
        super(GPF_plus_Prompt, self).__init__()
        # 论文定义：GPF+ 维护一组基向量 (basis vectors)
        self.prompt_basis = nn.Parameter(torch.empty(prompt_num, in_dim))
        nn.init.xavier_uniform_(self.prompt_basis)
        # 用一个线性层计算每个节点对不同基向量的注意力权重
        self.attn = nn.Linear(in_dim, prompt_num)

    def forward(self, x):
        # x: [N, in_dim]
        score = self.attn(x)  # [N, prompt_num]
        weight = F.softmax(score, dim=1)
        # 聚合基向量: [N, prompt_num] @ [prompt_num, in_dim] -> [N, in_dim]
        node_specific_prompt = torch.matmul(weight, self.prompt_basis)
        # 为每个节点加上属于它自己的专属 Prompt
        return x + node_specific_prompt