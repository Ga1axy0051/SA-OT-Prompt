import torch
import torch.nn as nn

class GraphPrompt_Prompt(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # GraphPrompt 学习一个与任务相关的全局提示向量，作用于 GNN 的输出表征上
        self.prompt_vector = nn.Parameter(torch.empty(1, in_dim))
        nn.init.xavier_uniform_(self.prompt_vector)

    def forward(self, node_embeddings):
        return node_embeddings + self.prompt_vector