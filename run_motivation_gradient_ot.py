import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 🚀 霸气锁卡设置 (在这里修改你抢到的卡号)
# ==========================================
GPU_ID = '2'  # 例如改成 '1', '2' 等
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前动机实验 1 正在运行的设备: {device} (物理卡号: {GPU_ID})")

torch.manual_seed(2026)

# ==========================================
# 严谨环境设定: 模拟真实的 Few-shot 场景
# ==========================================
N = 100       
dim = 128     
M = 20        
num_classes = 5

# 将生成的数据直接塞进指定的 GPU
X = torch.randn(N, dim, device=device)
X[:30] += 2.0  
X[30:60] -= 2.0
labels = torch.randint(0, num_classes, (N,), device=device)

class AttentionPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(torch.randn(M, dim))
        
    def forward(self, x):
        score = torch.matmul(x, self.P.t()) 
        attn = F.softmax(score, dim=-1)     
        x_prompted = x + torch.matmul(attn, self.P)
        return x_prompted, attn

class OTPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(torch.randn(M, dim))
        self.epsilon = 0.05
        
    def forward(self, x, max_iter=50):
        x_norm = F.normalize(x, p=2, dim=1)
        p_norm = F.normalize(self.P, p=2, dim=1)
        C = 1.0 - torch.matmul(x_norm, p_norm.t())
        
        K = torch.exp(-C / self.epsilon)
        # ⚠️ 关键修复：确保 OT 的中间变量也在同一张卡上
        u = torch.ones(N, requires_grad=False, device=x.device) / N
        v = torch.ones(M, requires_grad=False, device=x.device) / M
        
        for _ in range(max_iter):
            u = (torch.ones(N, device=x.device) / N) / torch.mv(K, v)
            v = (torch.ones(M, device=x.device) / M) / torch.mv(K.t(), u)
            
        T_ot = torch.diag(u) @ K @ torch.diag(v)
        x_prompted = x + torch.matmul(T_ot * N, self.P)
        return x_prompted, T_ot * N

# 实例化模型并推入指定 GPU
model_attn = AttentionPrompt().to(device)
model_ot = OTPrompt().to(device)
classifier_attn = nn.Linear(dim, num_classes).to(device)
classifier_ot = nn.Linear(dim, num_classes).to(device)

opt_attn = torch.optim.Adam(list(model_attn.parameters()) + list(classifier_attn.parameters()), lr=0.01)
opt_ot = torch.optim.Adam(list(model_ot.parameters()) + list(classifier_ot.parameters()), lr=0.01)

# ==========================================
# 训练循环：抓取真实的梯度范数
# ==========================================
epochs = 50
grad_norms_attn = []
grad_norms_ot = []

for epoch in range(epochs):
    # --- Attention 组 ---
    opt_attn.zero_grad()
    out_attn, matrix_attn = model_attn(X)
    loss_attn = F.cross_entropy(classifier_attn(out_attn), labels)
    loss_attn.backward()
    grad_norms_attn.append(model_attn.P.grad.norm().item())
    opt_attn.step()
    
    # --- OT 组 ---
    opt_ot.zero_grad()
    out_ot, matrix_ot = model_ot(X)
    loss_ot = F.cross_entropy(classifier_ot(out_ot), labels)
    loss_ot.backward()
    grad_norms_ot.append(model_ot.P.grad.norm().item())
    opt_ot.step()

# ==========================================
# 严谨绘图 (⚠️ 必须加 .cpu() 把数据拉回内存画图)
# ==========================================
fig = plt.figure(figsize=(18, 5))

ax1 = plt.subplot(1, 3, 1)
ax1.plot(grad_norms_attn, label="Attention (Gradient Truncation)", color="#d62728", linewidth=2.5)
ax1.plot(grad_norms_ot, label="OT (Robust Gradient)", color="#1f77b4", linewidth=2.5)
ax1.set_title("Prompt Gradient Norm during Fine-tuning", fontsize=13, fontweight='bold')
ax1.set_xlabel("Epochs", fontsize=11)
ax1.set_ylabel("L2 Norm of $\\nabla P$", fontsize=11)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = plt.subplot(1, 3, 2)
sns.heatmap(matrix_attn.cpu().detach().numpy(), cmap="Reds", cbar=False, ax=ax2)
ax2.set_title("Vanilla Attention Assignment\n(Severe Mode Collapse)", fontsize=13, fontweight='bold')
ax2.set_xlabel("Prompt Tokens", fontsize=11)
ax2.set_ylabel("Nodes", fontsize=11)

ax3 = plt.subplot(1, 3, 3)
sns.heatmap(matrix_ot.cpu().detach().numpy(), cmap="Blues", cbar=False, ax=ax3)
ax3.set_title("OT Assignment (Ours)\n(Global Manifold Alignment)", fontsize=13, fontweight='bold')
ax3.set_xlabel("Prompt Tokens", fontsize=11)

plt.tight_layout()
plt.savefig("rigorous_gradient_and_mode_collapse.pdf", dpi=300, bbox_inches='tight')
print("✅ 严谨动力学图表已生成: rigorous_gradient_and_mode_collapse.pdf")