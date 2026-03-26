import torch
import torch.nn as nn

class GPPT_Prompt(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(GPPT_Prompt, self).__init__()
        # GPPT 灵魂：每个类别分配一个可学习的 Task Token
        self.task_tokens = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.task_tokens)

    def forward(self, node_embeddings):
        # 绝对原汁原味：直接用节点特征和 Task Tokens 算内积，作为类别的 logits
        # node_embeddings: [N, in_dim], task_tokens: [C, in_dim]
        # output: [N, C]
        logits = torch.matmul(node_embeddings, self.task_tokens.t())
        return logits