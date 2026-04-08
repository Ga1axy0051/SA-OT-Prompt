import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import add_self_loops, dropout_adj
from torch_geometric.nn import GCNConv
import dgl  
import sys
import yaml
import importlib

# 导入自定义模块
from utils.data_loader import load_dataset, generate_few_shot_splits, inject_noise_edges
from models.ot_prompt import SAOTPrompt
from models.uniprompt import UniPrompt
from models.hybrid_prompt import HybridPrompt
from utils.legacy_utils import normalize_edge, NodeEva

# 导入基座模型
from models.graphmae import build_model
from models.DGI import DGI, DGI_process
from models.GRACE import Encoder, Model as GRACE_Model, drop_feature
from models.Base import LogReg

# 导入所有 Baseline 的 Prompt 模块
from models import GPPT_Prompt, GPF_Prompt, GPF_plus_Prompt, EdgePrompt, EdgePrompt_plus, GraphPrompt_Prompt, AllInOne_Prompt

def compute_homophily(edge_index, y):
    src, dst = edge_index
    same_label = (y[src] == y[dst]).sum().item()
    return same_label / edge_index.size(1) if edge_index.size(1) > 0 else 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def run(args, device):
    # ==============================================================
    # 1. 加载与预处理数据
    # ==============================================================
    data, input_dim, output_dim = load_dataset(args.dataset, data_dir="./data/raw")
    data = data.to(device)
    
    # 🟢 [核心修复 A]：全局索引越界校验与特征 Padding (异配图防御阵)
    max_node_idx = int(data.edge_index.max())
    if max_node_idx >= data.num_nodes:
        actual_num_nodes = max_node_idx + 1
        print(f"⚠️ [警告] {args.dataset} 索引越界! 声明: {data.num_nodes}, 实际最大: {max_node_idx}")
        print(f"🛠️ 正在扩充节点数至 {actual_num_nodes} 并补齐特征/标签...")
        
        if data.x.size(0) < actual_num_nodes:
            pad_size = actual_num_nodes - data.x.size(0)
            padding_x = torch.zeros((pad_size, data.x.size(1)), device=device)
            data.x = torch.cat([data.x, padding_x], dim=0)
            
        if data.y.size(0) < actual_num_nodes:
            pad_size = actual_num_nodes - data.y.size(0)
            padding_y = torch.full((pad_size,), -1, dtype=data.y.dtype, device=device)
            data.y = torch.cat([data.y, padding_y], dim=0)
            
        data.num_nodes = actual_num_nodes

    if args.feat_mask > 0:
        mask = torch.rand_like(data.x) > args.feat_mask
        data.x = data.x * mask

    if args.noise > 0:
        data.edge_index = inject_noise_edges(data.edge_index, data.y, args.noise)
    
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32, device=device)
    edge_index, edge_weight = add_self_loops(data.edge_index, edge_weight)
    edge_weight = normalize_edge(edge_index, edge_weight, data.num_nodes).to(device)
    edge_index = edge_index.to(device)

    # ==============================================================
    # 2. Stage 1: 加载 / 全自动训练预训练基座
    # ==============================================================
    model_save_path = f'./pretrain_model/{args.model}/{args.dataset}_hid{args.hid_dim}.pkl'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    if args.model == 'GraphMAE':
        raw_model = build_model(num_hidden=args.hid_dim, num_features=input_dim).to(device)
    
    elif args.model == 'DGI':
        raw_model = DGI(input_dim, args.hid_dim, 'prelu').to(device)
    
    elif args.model == 'GRACE':
        encoder = Encoder(input_dim, args.hid_dim, nn.PReLU(), base_model=GCNConv, k=2).to(device)
        raw_model = GRACE_Model(encoder, args.hid_dim, args.hid_dim, 0.5).to(device)
    
    elif args.model == 'GraphMAE2':
        from argparse import Namespace
        print(f"=== 正在加载 GraphMAE2 ({args.dataset}) SOTA预训练基座 ===")
        
        graphmae2_root = os.path.abspath('./pretrain_model/graphmae2')
        curr_dir = os.getcwd()
        old_sys_path = sys.path[:]
        modules_backup = {k: v for k, v in sys.modules.items() if k == 'models' or k.startswith('models.') or k == 'utils' or k.startswith('utils.')}
        for k in modules_backup: sys.modules.pop(k)
        
        new_path = [graphmae2_root]
        for p in old_sys_path:
            if os.path.abspath(p) != os.path.abspath(curr_dir) and p != '':
                new_path.append(p)
        sys.path = new_path
        
        try:
            import models as mae2_pkg
            importlib.reload(mae2_pkg)
            build_mae2 = mae2_pkg.build_model
            
            # 🟢 [路线 A 终极版：极致公平] 
            # 彻底抛弃官方 YAML，核心参数直接绑定你的 args (MAE1 用的啥，MAE2 就用啥)
            num_heads = 4
            dim_per_head = args.hid_dim // num_heads
            
            cfg = {
                'lr': args.lr,                 # 🔴 强制对齐 MAE1 的学习率
                'weight_decay': args.wd,       # 🔴 强制对齐 MAE1 的权重衰减
                'optimizer': 'adam',
                'num_hidden': args.hid_dim,  # 直接设为 256
                'num_heads': 4,              # 依然用 4 个头
                'concat_out': False,
                'num_out_heads': 1, 'num_layers': 2, 'num_dec_layers': 1, 
                
                # 下面是 GraphMAE2 独有的架构参数，保持其官方默认推荐即可
                'mask_rate': 0.5, 'mask_method': 'random', 
                'remask_rate': 0.5, 'remask_method': 'random', 
                'num_remasking': 3, 'replace_rate': 0.05, 'drop_edge_rate': 0.0,
                'encoder': 'gat', 'decoder': 'gat', 'activation': 'prelu', 
                'negative_slope': 0.2, 'attn_drop': 0.1, 'in_drop': 0.2, 
                'norm': 'layernorm', 'residual': True, 'concat_out': True, 
                'loss_fn': 'sce', 'alpha_l': 2.0, 'lam': 1.0,
                'delayed_ema_epoch': 0, 'momentum': 0.996, 
                'type_grad': 'grad', 'pooling': 'mean', 'zero_init': False, 'is_sparse': False,
            }
            
            # ❌ 删除了所有关于 yaml 读取的代码 ❌
            # 现在的 cfg 是绝对受我们掌控的纯净字典
            
            cfg['num_features'] = input_dim 
            cfg['dataset'] = args.dataset
            mae2_args = Namespace(**cfg)
            raw_model = build_mae2(mae2_args).to(device)
            print(f">> 基座架构加载成功 (Total Dim: {args.hid_dim}, LR: {args.lr})")
            
        finally:
            for k in list(sys.modules.keys()):
                if k == 'models' or k.startswith('models.'): sys.modules.pop(k)
            sys.modules.update(modules_backup)
            sys.path = old_sys_path

        # 权重加载/即时预训练
        ckpt_dir = os.path.join(graphmae2_root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        mae2_ckpt_path = os.path.join(ckpt_dir, f"{args.dataset}.pt")
        
        try:
            raw_model.load_state_dict(torch.load(mae2_ckpt_path, map_location=device))
            print(f">> 成功挂载权重: {args.dataset}.pt")
        except:
            print(f">> 🚀 权重缺失或维度不匹配，启动 GraphMAE2 现场【即时预训练】...")
            raw_model.train()
            
            # 🟢 直接使用 cfg 里已经绑定好的 args.lr 和 args.wd
            optimizer_pt = torch.optim.Adam(
                raw_model.parameters(), 
                lr=cfg['lr'], 
                weight_decay=cfg['weight_decay']
            )
            
            src, dst = edge_index
            g_pt = dgl.graph((src, dst), num_nodes=data.num_nodes).to(device)
            g_pt = dgl.add_self_loop(dgl.remove_self_loop(g_pt))
            for pt_epoch in range(args.epochs): 
                loss_pt = raw_model(g_pt, data.x)
                optimizer_pt.zero_grad(); loss_pt.backward(); optimizer_pt.step()
                if (pt_epoch + 1) % 100 == 0:
                    print(f"   - Base Epoch {pt_epoch+1}/{args.epochs} | Loss: {loss_pt.item():.4f}")
            torch.save(raw_model.state_dict(), mae2_ckpt_path)
            model_save_path = mae2_ckpt_path # 统一路径记录

        raw_model.eval()
        for param in raw_model.encoder.parameters(): param.requires_grad = False

    # [2] 老模型预训练引擎 (DGI/GRACE/GraphMAE v1)
    if not os.path.exists(model_save_path) and args.model != 'GraphMAE2':
        print(f"🔥 未找到 {args.model} 权重，正在全自动执行预训练...")
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=args.lr, weight_decay=args.wd)
        if args.model == 'DGI': loss_func = nn.BCEWithLogitsLoss()
        raw_model.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            if args.model == 'GraphMAE': loss, _ = raw_model(data.x, edge_index, edge_weight)
            elif args.model == 'DGI':
                shuf_x, lbl = DGI_process(data.num_nodes, data.x)
                shuf_x, lbl = shuf_x.to(device), lbl.to(device)
                logits = raw_model(data.x, shuf_x, edge_index, edge_weight, None, None, None)
                loss = loss_func(logits, lbl)
            elif args.model == 'GRACE':
                edge_index_1, edge_weight_1 = dropout_adj(edge_index, edge_weight, p=0.2)
                edge_index_2, edge_weight_2 = dropout_adj(edge_index, edge_weight, p=0.2)
                x_1, x_2 = drop_feature(data.x, 0.2), drop_feature(data.x, 0.2)
                z1, z2 = raw_model(x_1, edge_index_1, edge_weight_1), raw_model(x_2, edge_index_2, edge_weight_2)
                loss = raw_model.loss(z1, z2, batch_size=0)
            loss.backward(); optimizer.step()
        torch.save(raw_model.state_dict(), model_save_path)
    
    if args.model != 'GraphMAE2':
        raw_model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

    # [3] 透明代理层
    class BackboneWrapper:
        def __init__(self, model_type, model):
            self.model_type, self.model = model_type, model
        def embed(self, x, edge_index, edge_weight=None):
            if self.model_type == 'DGI':
                z, _ = self.model.embed(x, edge_index, edge_weight, None)
                return z
            elif self.model_type == 'GraphMAE2':
                src, dst = edge_index
                g_eval = dgl.graph((src, dst), num_nodes=x.shape[0]).to(x.device)
                g_eval = dgl.add_self_loop(dgl.remove_self_loop(g_eval))
                return self.model.embed(g_eval, x)
            return self.model.embed(x, edge_index, edge_weight)
        def parameters(self):
            return self.model.encoder.parameters() if self.model_type == 'GraphMAE2' else self.model.parameters()
        def state_dict(self):
            return self.model.encoder.state_dict() if self.model_type == 'GraphMAE2' else self.model.state_dict()
        def load_state_dict(self, sd):
            if self.model_type == 'GraphMAE2': self.model.encoder.load_state_dict(sd)
            else: self.model.load_state_dict(sd)
        def __getattr__(self, name): return getattr(self.model, name)

    base_encoder = BackboneWrapper(args.model, raw_model)

    # ==============================================================
    # 3. Stage 2: 下游 Few-shot 适配 (全 Baseline 完整版)
    # ==============================================================
    test_accs = []
    down_loss_fn = nn.CrossEntropyLoss()

    for trail in range(1, args.trails + 1):
        current_seed = args.seed + trail 
        data = generate_few_shot_splits(data, output_dim, shot=args.shot, seed=current_seed)
        classifier = LogReg(args.hid_dim, output_dim).to(device)
        prompt = None 
        
        # --- 初始化判定 ---
        if args.method == 'sa_ot_prompt':
            prompt = SAOTPrompt(data.x, input_dim, args.num_prompts, args.ot_epsilon, args.k).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": prompt.parameters(), "lr": args.down_lr}, {"params": classifier.parameters(), "lr": args.clf_lr}], weight_decay=args.down_wd)
            
        elif args.method == 'linear_probe':
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam(classifier.parameters(), lr=args.clf_lr, weight_decay=args.down_wd)
            
        elif args.method == 'fine_tune':
            base_encoder.train(); [p.requires_grad_(True) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": base_encoder.parameters(), "lr": args.down_lr}, {"params": classifier.parameters(), "lr": args.clf_lr}], weight_decay=args.down_wd)
        
        elif args.method == 'uniprompt':
            prompt = UniPrompt(x=data.x, k=args.k, metric='cosine', alpha=1.0, num_nodes=data.num_nodes).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": prompt.parameters(), "lr": args.down_lr}, {"params": classifier.parameters(), "lr": args.clf_lr}], weight_decay=args.down_wd)

        elif args.method == 'gppt':
            classifier = GPPT_Prompt(in_dim=args.hid_dim, num_classes=output_dim).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam(classifier.parameters(), lr=args.down_lr, weight_decay=args.down_wd)
            
        elif args.method in ['gpf', 'gpf_plus', 'all_in_one']:
            if args.method == 'gpf': prompt = GPF_Prompt(in_dim=input_dim).to(device)
            elif args.method == 'gpf_plus': prompt = GPF_plus_Prompt(in_dim=input_dim).to(device)
            elif args.method == 'all_in_one': prompt = AllInOne_Prompt(in_dim=input_dim).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": prompt.parameters(), "lr": args.down_lr}, {"params": classifier.parameters(), "lr": args.clf_lr}], weight_decay=args.down_wd)

        elif args.method in ['edgeprompt', 'edgeprompt_plus']:
            if args.method == 'edgeprompt': prompt = EdgePrompt(in_dim=input_dim).to(device)
            else: prompt = EdgePrompt_plus(in_dim=input_dim).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": prompt.parameters(), "lr": args.down_lr}, {"params": classifier.parameters(), "lr": args.clf_lr}], weight_decay=args.down_wd)

        elif args.method == 'graphprompt':
            prompt = GraphPrompt_Prompt(in_dim=args.hid_dim).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": prompt.parameters(), "lr": args.down_lr}, {"params": classifier.parameters(), "lr": args.clf_lr}], weight_decay=args.down_wd)
        
        elif args.method == 'hybrid_prompt':
            prompt = HybridPrompt(data.x, input_dim, args.num_prompts, args.ot_epsilon, args.k, args.alpha).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": prompt.parameters(), "lr": args.down_lr}, {"params": classifier.parameters(), "lr": args.clf_lr}], weight_decay=args.down_wd)

        # --- 训练循环 ---
        best_val_acc = -1.0
        cnt_wait = 0
        best_classifier_state, best_prompt_state, best_base_state = None, None, None

        for epoch in range(args.down_epochs):
            if prompt is not None: prompt.train()
            classifier.train()
            optimizer_down.zero_grad()
            ot_loss_val = torch.tensor(0.0).to(device)

            if args.method in ['sa_ot_prompt', 'hybrid_prompt']:
                x_ad, ot_l, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
                c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(x_ad, c_idx, c_w)
                ot_loss_val = ot_l
            elif args.method == 'uniprompt':
                pt_idx, pt_w = prompt()
                c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(data.x, c_idx, c_w)
            elif args.method in ['gpf', 'gpf_plus']:
                embeds = base_encoder.embed(prompt(data.x), edge_index, edge_weight)
            elif args.method == 'all_in_one':
                new_x, ni, nw = prompt(data.x, edge_index, edge_weight)
                embeds = base_encoder.embed(new_x, ni, nw)[:data.num_nodes]
            elif args.method in ['edgeprompt', 'edgeprompt_plus']:
                embeds = base_encoder.embed(prompt(data.x, edge_index), edge_index, edge_weight)
            elif args.method == 'graphprompt':
                embeds = prompt(base_encoder.embed(data.x, edge_index, edge_weight))
            else:
                embeds = base_encoder.embed(data.x, edge_index, edge_weight)
            
            logits = classifier(embeds)
            loss = down_loss_fn(logits[data.train_mask], data.y[data.train_mask]) + (args.ot_beta * ot_loss_val if args.method == 'sa_ot_prompt' else 0)
            loss.backward(); optimizer_down.step()

            with torch.no_grad():
                classifier.eval()
                v_logits = classifier(embeds)
                v_acc = (v_logits[data.val_mask].argmax(1) == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
                if v_acc > best_val_acc:
                    best_val_acc, cnt_wait = v_acc, 0
                    best_classifier_state = {k: v.cpu() for k, v in classifier.state_dict().items()}
                    if prompt is not None: best_prompt_state = {k: v.cpu() for k, v in prompt.state_dict().items()}
                    if args.method == 'fine_tune': best_base_state = {k: v.cpu() for k, v in base_encoder.state_dict().items()}
                else:
                    cnt_wait += 1
                    if cnt_wait >= args.patience: break

        # --- 测试评估 ---
        classifier.load_state_dict({k: v.to(device) for k, v in best_classifier_state.items()}); classifier.eval()
        if prompt is not None: prompt.load_state_dict({k: v.to(device) for k, v in best_prompt_state.items()}); prompt.eval()
        with torch.no_grad():
            if args.method in ['sa_ot_prompt', 'hybrid_prompt']:
                x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
                c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(x_ad, c_idx, c_w)
            elif args.method == 'uniprompt':
                pt_idx, pt_w = prompt()
                c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(data.x, c_idx, c_w)
            elif args.method in ['gpf', 'gpf_plus']:
                embeds = base_encoder.embed(prompt(data.x), edge_index, edge_weight)
            elif args.method == 'all_in_one':
                nx, ni, nw = prompt(data.x, edge_index, edge_weight)
                embeds = base_encoder.embed(nx, ni, nw)[:data.num_nodes]
            elif args.method in ['edgeprompt', 'edgeprompt_plus']:
                embeds = base_encoder.embed(prompt(data.x, edge_index), edge_index, edge_weight)
            elif args.method == 'graphprompt':
                embeds = prompt(base_encoder.embed(data.x, edge_index, edge_weight))
            else:
                if args.method == 'fine_tune': base_encoder.load_state_dict({k: v.to(device) for k, v in best_base_state.items()})
                embeds = base_encoder.embed(data.x, edge_index, edge_weight)
            
            t_acc, _, _, _ = NodeEva(classifier(embeds), torch.nonzero(data.test_mask).squeeze(), data, output_dim, device)
            test_accs.append(t_acc)
        torch.cuda.empty_cache()

    print(f"[{args.model}] + [{args.method}] Final Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--model', type=str, default='GraphMAE', choices=['GraphMAE', 'DGI', 'GRACE', 'GraphMAE2']) # 补充了 GraphMAE2
    parser.add_argument('--method', type=str, default='sa_ot_prompt', 
                        choices=['sa_ot_prompt', 'linear_probe', 'fine_tune', 'uniprompt', 
                                 'hybrid_prompt', 'gppt', 'gpf', 'gpf_plus', 
                                 'graphprompt', 'all_in_one', 'edgeprompt', 'edgeprompt_plus'])
    parser.add_argument('--seed', type=int, default=42)
    
    # ==============================================================
    # 🛡️ 绝对锁死区 (The Golden Baseline) - 任何实验不得更改！
    # ==============================================================
    parser.add_argument('--hid_dim', type=int, default=256)        # 🟢 锁死 256：匹配预训练权重
    parser.add_argument('--num_prompts', type=int, default=10)     # 🟢 锁死 10：对齐特征空间容量
    parser.add_argument('--clf_lr', type=float, default=0.05)      # 🟢 锁死 0.05：强行激活分类器
    parser.add_argument('--down_epochs', type=int, default=2000)   # 🟢 锁死 2000：彻底放开训练上限
    parser.add_argument('--patience', type=int, default=100)       # 🟢 锁死 100：抗震荡
    parser.add_argument('--trails', type=int, default=30)          # 🟢 锁死 30：论文级标准差
    parser.add_argument('--down_wd', type=float, default=5e-5)     # 🟢 锁死 5e-5：固定正则化
    
    # 预训练参数 (如果找不到权重，触发临时预训练时的备用参数)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=500)
    
    # ==============================================================
    # 🔍 动态搜索区 (由 tuner.py 传入)
    # ==============================================================
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--down_lr', type=float, default=0.005)    # 动态：Prompt 学习率
    parser.add_argument('--tau', type=float, default=0.5)          # 动态：边融合权重
    parser.add_argument('--k', type=int, default=50)               # 动态：子图采样数
    parser.add_argument('--ot_beta', type=float, default=0.01)     # 动态：最优传输权重
    
    # 其他默认参数
    parser.add_argument('--ot_epsilon', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--feat_mask', type=float, default=0.0)
    
    args = parser.parse_args()
    set_seed(args.seed)
    run(args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))