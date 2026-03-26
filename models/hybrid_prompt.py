import torch
import torch.nn as nn
from models.ot_prompt import SAOTPrompt
from models.uniprompt import UniPrompt

class HybridPrompt(nn.Module):
    def __init__(self, x, input_dim, num_prompts, ot_epsilon=0.1, k=50, alpha=0.5):
        """
        NIPS 级双流图提示框架 (Module Ensembling)
        绝对安全：直接封装原版模型，零底层干预！
        """
        super(HybridPrompt, self).__init__()
        self.alpha = alpha
        
        # 👑 直接把两台原装发动机装进来！
        self.sa_ot = SAOTPrompt(x, input_dim, num_prompts, ot_epsilon, k)
        # 传入 x.size(0) 作为 num_nodes
        self.uni = UniPrompt(x=x, k=k, metric='cosine', alpha=1.0, num_nodes=x.size(0))

    def forward(self, x, edge_index, edge_weight):
        # 1. 跑你的原装 SA-OT 引擎 (拿到全局流形结构)
        x_ad, ot_loss, pt_idx_ot, pt_w_ot = self.sa_ot(x, edge_index, edge_weight)

        # 2. 跑原装的 UniPrompt 引擎 (拿到局部特征近邻)
        pt_idx_uni, pt_w_uni = self.uni()

        # 3. 终极 Late Fusion (后期融合)
        # 把两组边拼在一起
        pt_idx_final = torch.cat([pt_idx_ot, pt_idx_uni], dim=1)
        
        # 用 alpha 动态分配权重 (alpha=0.5 就是五五开)
        pt_w_final = torch.cat([pt_w_ot * (1.0 - self.alpha), pt_w_uni * self.alpha], dim=0)

        return x_ad, ot_loss, pt_idx_final, pt_w_final

    def edge_fuse(self, orig_idx, orig_w, pt_idx, pt_w, tau):
        # 融合图结构的逻辑也直接借用你原版 SA-OT 的，绝对不造轮子
        return self.sa_ot.edge_fuse(orig_idx, orig_w, pt_idx, pt_w, tau)