import torch.nn as nn

class LogReg(nn.Module):
    """
    下游任务的简单逻辑回归分类头 (Logistic Regression Classifier)
    极其干净，没有任何冗余依赖！
    """
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        
        # 顶会细节：良好的初始化有助于 Few-shot 快速收敛
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)