import torch
import torch.nn as nn

class SinkhornOT(nn.Module):
    """
    可微熵正则化最优传输层 (Differentiable Entropic Optimal Transport Layer)
    """
    def __init__(self, epsilon=0.1, max_iters=20):
        super(SinkhornOT, self).__init__()
        self.epsilon = epsilon
        self.max_iters = max_iters

    def forward(self, x, p):
        # 1. 快速计算成对欧氏距离平方
        dist_x = torch.sum(x ** 2, dim=1, keepdim=True)
        dist_p = torch.sum(p ** 2, dim=1, keepdim=True).t()
        C = dist_x + dist_p - 2.0 * torch.matmul(x, p.t())
        
        # 2. 代价矩阵归一化 (防指数下溢出的核心数值稳定技巧)
        C = C / (C.max() + 1e-8)
        K_eps = torch.exp(-C / self.epsilon)

        # 3. 初始化边缘分布 (均匀分布)
        N, K = x.size(0), p.size(0)
        mu = torch.empty(N, dtype=x.dtype, device=x.device).fill_(1.0 / N)
        nu = torch.empty(K, dtype=x.dtype, device=x.device).fill_(1.0 / K)

        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # 4. Sinkhorn-Knopp 交替迭代 (完全可导)
        for _ in range(self.max_iters):
            u = mu / (torch.matmul(K_eps, v) + 1e-8)
            v = nu / (torch.matmul(K_eps.t(), u) + 1e-8)

        # 5. 计算最终的最优传输计划 T* 和 OT 损失
        T_star = u.unsqueeze(1) * K_eps * v.unsqueeze(0)
        ot_loss = torch.sum(T_star * C)

        return T_star, ot_loss