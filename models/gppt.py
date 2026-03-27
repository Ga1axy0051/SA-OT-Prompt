import torch
import torch.nn as nn

class GPPT_Prompt(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(GPPT_Prompt, self).__init__()
        # GPPT 灵魂：为每个类别学习一个代表性的 Task Token
        self.task_tokens = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.task_tokens)

    def forward(self, node_embeddings):
        # 绝对原汁原味：节点 Embedding 和 Task Tokens 的内积 (Dot-product)
        # node_embeddings: [N, in_dim]
        # task_tokens: [num_classes, in_dim]
        # 输出 logits: [N, num_classes]
        logits = torch.matmul(node_embeddings, self.task_tokens.t())
        return logits