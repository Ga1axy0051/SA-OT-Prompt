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
import dgl  # 🟢 引入 DGL 供 GraphMAE2 使用
import sys
import yaml
import importlib

# 导入自定义模块
from utils.data_loader import load_dataset, generate_few_shot_splits, inject_noise_edges
from models.ot_prompt import SAOTPrompt
from models.uniprompt import UniPrompt
from models.hybrid_prompt import HybridPrompt
from utils.legacy_utils import normalize_edge, NodeEva

# 导入基座模型 (已包含 DGI 和 GRACE 的组件)
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

    # [1] 实例化底座
    if args.model == 'GraphMAE':
        # 🟢 原版 GraphMAE，完全保留你的旧逻辑
        raw_model = build_model(num_hidden=args.hid_dim, num_features=input_dim).to(device)
    
    elif args.model == 'DGI':
        raw_model = DGI(input_dim, args.hid_dim, 'prelu').to(device)
    
    elif args.model == 'GRACE':
        encoder = Encoder(input_dim, args.hid_dim, nn.PReLU(), base_model=GCNConv, k=2).to(device)
        raw_model = GRACE_Model(encoder, args.hid_dim, args.hid_dim, 0.5).to(device)
    
    elif args.model == 'GraphMAE2':
        import sys
        import importlib
        from argparse import Namespace
        
        print(f"=== 正在加载 GraphMAE2 ({args.dataset}) SOTA预训练基座 ===")
        
        # 1. 🌟 绝对物理路径定位
        graphmae2_root = os.path.abspath('./pretrain_model/graphmae2')
        curr_dir = os.getcwd()

        # 2. 🌟 【物理隔绝】备份并彻底清洗环境
        # 备份 sys.path 和 sys.modules
        old_sys_path = sys.path[:]
        modules_backup = {k: v for k, v in sys.modules.items() if k == 'models' or k.startswith('models.') or k == 'utils' or k.startswith('utils.')}
        
        # 彻底移除当前目录和所有 models 相关缓存
        for k in modules_backup: sys.modules.pop(k)
        
        # 强行让 sys.path 第一位是 GraphMAE2，并且移除当前主目录 '.'
        # 这样 Python 在 import models 时，绝对看不见你主项目的 models
        new_path = [graphmae2_root]
        for p in old_sys_path:
            if os.path.abspath(p) != os.path.abspath(curr_dir) and p != '':
                new_path.append(p)
        sys.path = new_path
        
        try:
            # 3. 🌟 核心加载：此时 Python 处于“失忆状态”，只会加载 GraphMAE2 的 models
            import models as mae2_pkg
            importlib.reload(mae2_pkg)
            build_mae2 = mae2_pkg.build_model
            
            # --- 智能配置加载 --- (逻辑保持不变)
            config_path = os.path.join(graphmae2_root, "configs", f"{args.dataset}.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = yaml.load(f, yaml.FullLoader)
            else:
                print(f">> [提示] 未找到 {args.dataset}.yaml，启动【异配图专属默认配置】")
                # 为异配小图（Cornell, Texas, Wisconsin等）量身定制的 SOTA 级超参配置
            # 🟢 GraphMAE2 全参数补全版：适配异配图/无 yaml 场景
            cfg = {
                # 1. 训练与优化
                'lr': 0.001, 
                'weight_decay': 1e-4, 
                'optimizer': 'adam',
                
                # 2. 架构维度
                'num_hidden': 512, 
                'num_heads': 4,
                'num_out_heads': 1, 
                'num_layers': 2, 
                'num_dec_layers': 1, 
                
                # 3. 掩码机制 (修复报错的核心)
                'mask_rate': 0.5,
                'mask_method': 'random',    # 🟢 修复当前报错：第一阶段掩码方式
                'remask_rate': 0.5, 
                'remask_method': 'random',  # 第二阶段重掩码方式
                'num_remasking': 3, 
                'replace_rate': 0.05, 
                'drop_edge_rate': 0.0,
                
                # 4. 编码器/解码器细节
                'encoder': 'gat', 
                'decoder': 'gat', 
                'activation': 'prelu', 
                'negative_slope': 0.2, 
                'attn_drop': 0.1,
                'in_drop': 0.2, 
                'norm': 'layernorm',
                'residual': True, 
                'concat_out': False,
                
                # 5. 损失函数与动量更新
                'loss_fn': 'sce', 
                'alpha_l': 2.0, 
                'lam': 1.0,
                'delayed_ema_epoch': 0, 
                'momentum': 0.996,         # 教师网络更新动量
                
                # 6. 杂项 (MAE2 构造函数会扫描的所有字段)
                'type_grad': 'grad', 
                'pooling': 'mean',
                'zero_init': False,
                'alpha_l': 2.0,            # 重复确认 SCE loss 参数
                'is_sparse': False,        # 是否处理稀疏图
            }
            
            cfg['num_features'] = input_dim 
            cfg['dataset'] = args.dataset
            cfg['residual'] = cfg.get('residual', False)
            cfg['zero_init'] = False
            mae2_args = Namespace(**cfg)
            
            # 4. 🌟 构建模型
            raw_model = build_mae2(mae2_args).to(device)
            print(f">> 基座架构加载成功！")
            
        finally:
            # 5. 🌟 【时空复原】把主项目的环境还回来
            # 移除 GraphMAE2 的临时 models 缓存
            for k in list(sys.modules.keys()):
                if k == 'models' or k.startswith('models.'): sys.modules.pop(k)
            
            # 恢复备份，恢复路径
            sys.modules.update(modules_backup)
            sys.path = old_sys_path

        # 6. 权重加载与 JIT 预训练 (逻辑保持不变)
        ckpt_dir = os.path.join(graphmae2_root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        model_save_path = os.path.join(ckpt_dir, f"{args.dataset}.pt")
        
        if os.path.exists(model_save_path):
            raw_model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f">> 成功挂载权重: {args.dataset}.pt")
        else:
            print(f">> 🚀 正在启动 GraphMAE2 本地【即时预训练】引擎...")
            raw_model.train()
            optimizer_pt = torch.optim.Adam(raw_model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            
            import dgl
            src, dst = edge_index
            g_pt = dgl.graph((src, dst), num_nodes=data.num_nodes).to(device)
            g_pt = dgl.add_self_loop(dgl.remove_self_loop(g_pt))
            
            for pt_epoch in range(500): 
                loss_pt = raw_model(g_pt, data.x)
                optimizer_pt.zero_grad(); loss_pt.backward(); optimizer_pt.step()
                if (pt_epoch + 1) % 100 == 0:
                    print(f"   - Epoch {pt_epoch+1}/500 | Loss: {loss_pt.item():.4f}")
            torch.save(raw_model.state_dict(), model_save_path)

        raw_model.eval()
        for param in raw_model.encoder.parameters(): param.requires_grad = False
        args.hid_dim = mae2_args.num_hidden

    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # [2] 老模型的全自动预训练引擎 (GraphMAE2 已在上面处理完毕，这里会跳过)
    if not os.path.exists(model_save_path) and args.model != 'GraphMAE2':
        print(f"🔥 未找到 {args.model} 权重 ({model_save_path})，正在全自动执行预训练...")
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        if args.model == 'DGI':
            loss_func = nn.BCEWithLogitsLoss()
            
        raw_model.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            
            if args.model == 'GraphMAE':
                loss, _ = raw_model(data.x, edge_index, edge_weight)
                
            elif args.model == 'DGI':
                shuf_x, lbl = DGI_process(data.num_nodes, data.x)
                shuf_x, lbl = shuf_x.to(device), lbl.to(device)
                logits = raw_model(data.x, shuf_x, edge_index, edge_weight, None, None, None)
                loss = loss_func(logits, lbl)
                
            elif args.model == 'GRACE':
                edge_index_1, edge_weight_1 = dropout_adj(edge_index, edge_weight, p=0.2)
                edge_index_2, edge_weight_2 = dropout_adj(edge_index, edge_weight, p=0.2)
                x_1 = drop_feature(data.x, 0.2)
                x_2 = drop_feature(data.x, 0.2)
                z1 = raw_model(x_1, edge_index_1, edge_weight_1)
                z2 = raw_model(x_2, edge_index_2, edge_weight_2)
                loss = raw_model.loss(z1, z2, batch_size=0)
                
            loss.backward()
            optimizer.step()
            
        torch.save(raw_model.state_dict(), model_save_path)
        print(f"✅ {args.model} 在 {args.dataset} 上的预训练已完成，权重已保存！")
    
    # [3] 统一加载已存在的/刚刚训练好的权重
    if args.model != 'GraphMAE2':
        raw_model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

    # [4] 🟢 终极透明代理层 (BackboneWrapper)：抹平 PyG 和 DGL 的差异
    class BackboneWrapper:
        def __init__(self, model_type, model):
            self.model_type = model_type
            self.model = model
            
        def embed(self, x, edge_index, edge_weight=None):
            if self.model_type == 'DGI':
                z, _ = self.model.embed(x, edge_index, edge_weight, None)
                return z
            elif self.model_type == 'GraphMAE2':
                import dgl
                src, dst = edge_index
                g_eval = dgl.graph((src, dst), num_nodes=x.shape[0]).to(x.device)
                g_eval = dgl.add_self_loop(dgl.remove_self_loop(g_eval))
                # 🟢 关键：GraphMAE2 必须调用其 PreModel 的 embed 方法
                return self.model.embed(g_eval, x)
            else:
                return self.model.embed(x, edge_index, edge_weight)
                
        # 让 optimizer 也能正确抓取参数
        def parameters(self):
            if self.model_type == 'GraphMAE2':
                return self.model.encoder.parameters()
            return self.model.parameters()
            
        def state_dict(self):
            if self.model_type == 'GraphMAE2':
                return self.model.encoder.state_dict()
            return self.model.state_dict()
            
        def load_state_dict(self, state_dict):
            if self.model_type == 'GraphMAE2':
                self.model.encoder.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
                
        def __getattr__(self, name):
            # 将其他未实现的方法转发给内部的 model
            return getattr(self.model, name)

    base_encoder = BackboneWrapper(args.model, raw_model)

    # ==============================================================
    # 3. Stage 2: 下游 Few-shot 适配 (从此高枕无忧)
    # ==============================================================
    test_accs, f1s, rocs = [], [], []
    down_loss_fn = nn.CrossEntropyLoss()

    for trail in range(1, args.trails + 1):
        current_seed = args.seed + trail 
        data = generate_few_shot_splits(data, output_dim, shot=args.shot, seed=current_seed)
        
        classifier = LogReg(args.hid_dim, output_dim).to(device)
        prompt = None 
        
        # --- 初始化 ---
        if args.method == 'sa_ot_prompt':
            prompt = SAOTPrompt(data.x, input_dim, args.num_prompts, args.ot_epsilon, args.k).to(device)
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": args.clf_lr}
            ], weight_decay=args.down_wd)
            
        elif args.method == 'linear_probe':
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam(classifier.parameters(), lr=args.clf_lr, weight_decay=args.down_wd)
            
        elif args.method == 'fine_tune':
            base_encoder.train()
            for param in base_encoder.parameters(): param.requires_grad = True
            optimizer_down = torch.optim.Adam([
                {"params": base_encoder.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": args.clf_lr}
            ], weight_decay=args.down_wd)
        
        elif args.method == 'uniprompt':
            prompt = UniPrompt(x=data.x, k=args.k, metric='cosine', alpha=1.0, num_nodes=data.num_nodes).to(device)
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": args.clf_lr}
            ], weight_decay=args.down_wd)

        elif args.method == 'gppt':
            classifier = GPPT_Prompt(in_dim=args.hid_dim, num_classes=output_dim).to(device)
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam(classifier.parameters(), lr=args.down_lr, weight_decay=args.down_wd)
            
        elif args.method in ['gpf', 'gpf_plus', 'all_in_one']:
            if args.method == 'gpf': prompt = GPF_Prompt(in_dim=input_dim).to(device)
            elif args.method == 'gpf_plus': prompt = GPF_plus_Prompt(in_dim=input_dim).to(device)
            elif args.method == 'all_in_one': prompt = AllInOne_Prompt(in_dim=input_dim).to(device)
            
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": args.clf_lr}
            ], weight_decay=args.down_wd)

        elif args.method in ['edgeprompt', 'edgeprompt_plus']:
            if args.method == 'edgeprompt': prompt = EdgePrompt(in_dim=input_dim).to(device)
            else: prompt = EdgePrompt_plus(in_dim=input_dim).to(device)
            
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": args.clf_lr}
            ], weight_decay=args.down_wd)

        elif args.method == 'graphprompt':
            prompt = GraphPrompt_Prompt(in_dim=args.hid_dim).to(device)
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": args.clf_lr}
            ], weight_decay=args.down_wd)
        
        elif args.method == 'hybrid_prompt':
            prompt = HybridPrompt(data.x, input_dim, args.num_prompts, args.ot_epsilon, args.k, args.alpha).to(device)
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": args.clf_lr}
            ], weight_decay=args.down_wd)

        best_val_acc = -1.0
        best_val_loss = 1e9
        cnt_wait = 0
        best_prompt_state, best_classifier_state, best_base_state = None, None, None

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
                comb_index, comb_weight = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(data.x, comb_index, comb_weight)
            
            elif args.method in ['gpf', 'gpf_plus']:
                prompted_x = prompt(data.x)
                embeds = base_encoder.embed(prompted_x, edge_index, edge_weight)
            
            elif args.method == 'all_in_one':
                new_x, new_edge_index, new_edge_weight = prompt(data.x, edge_index, edge_weight)
                full_embeds = base_encoder.embed(new_x, new_edge_index, new_edge_weight)
                embeds = full_embeds[:data.num_nodes]
            
            elif args.method in ['edgeprompt', 'edgeprompt_plus']:
                prompted_x = prompt(data.x, edge_index)
                embeds = base_encoder.embed(prompted_x, edge_index, edge_weight)
            
            elif args.method == 'graphprompt':
                raw_embeds = base_encoder.embed(data.x, edge_index, edge_weight)
                embeds = prompt(raw_embeds)
            
            else: # fine_tune, linear_probe, gppt
                embeds = base_encoder.embed(data.x, edge_index, edge_weight)
            
            logits = classifier(embeds)
            train_loss = down_loss_fn(logits[data.train_mask], data.y[data.train_mask])
            
            loss = train_loss + (args.ot_beta * ot_loss_val if args.method == 'sa_ot_prompt' else 0)
            loss.backward()
            optimizer_down.step()

            # --- 验证逻辑 ---
            with torch.no_grad():
                classifier.eval()
                val_logits = classifier(embeds)
                v_loss = down_loss_fn(val_logits[data.val_mask], data.y[data.val_mask])
                v_acc = (val_logits[data.val_mask].argmax(1) == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

            if v_acc > best_val_acc or (v_acc == best_val_acc and v_loss.item() < best_val_loss):
                best_val_acc = v_acc
                best_val_loss = v_loss.item()
                cnt_wait = 0
                best_classifier_state = {k: v.cpu() for k, v in classifier.state_dict().items()}
                if prompt is not None: best_prompt_state = {k: v.cpu() for k, v in prompt.state_dict().items()}
                if args.method == 'fine_tune': best_base_state = {k: v.cpu() for k, v in base_encoder.state_dict().items()}
            else:
                cnt_wait += 1
                if cnt_wait >= args.patience: break

        # 4. 测试评估
        classifier.load_state_dict({k: v.to(device) for k, v in best_classifier_state.items()})
        classifier.eval()
        with torch.no_grad():
            if prompt is not None:
                prompt.load_state_dict({k: v.to(device) for k, v in best_prompt_state.items()})
                prompt.eval()

            if args.method in ['sa_ot_prompt', 'hybrid_prompt']:
                x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
                comb_index, comb_weight = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(x_ad, comb_index, comb_weight)
            
            elif args.method == 'uniprompt':
                pt_idx, pt_w = prompt()
                comb_index, comb_weight = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(data.x, comb_index, comb_weight)
            
            elif args.method in ['gpf', 'gpf_plus']:
                prompted_x = prompt(data.x)
                embeds = base_encoder.embed(prompted_x, edge_index, edge_weight)
                
            elif args.method == 'all_in_one':
                new_x, new_edge_index, new_edge_weight = prompt(data.x, edge_index, edge_weight)
                full_embeds = base_encoder.embed(new_x, new_edge_index, new_edge_weight)
                embeds = full_embeds[:data.num_nodes]
                
            elif args.method in ['edgeprompt', 'edgeprompt_plus']:
                prompted_x = prompt(data.x, edge_index)
                embeds = base_encoder.embed(prompted_x, edge_index, edge_weight)
                
            elif args.method == 'graphprompt':
                raw_embeds = base_encoder.embed(data.x, edge_index, edge_weight)
                embeds = prompt(raw_embeds)

            else:
                if args.method == 'fine_tune' and best_base_state is not None: 
                    base_encoder.load_state_dict({k: v.to(device) for k, v in best_base_state.items()})
                base_encoder.eval()
                embeds = base_encoder.embed(data.x, edge_index, edge_weight)

            test_logits = classifier(embeds)
            test_idx = torch.nonzero(data.test_mask).squeeze()
            t_acc, t_f1, t_roc, t_prc = NodeEva(test_logits, test_idx, data, output_dim, device)
            
            test_accs.append(t_acc)
            f1s.append(t_f1)

            del classifier, optimizer_down
            if prompt is not None: del prompt
            torch.cuda.empty_cache()

    print(f"[{args.model}] + [{args.method}] Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    # 🟢 注意：这里加入了 GraphMAE2 供你选择！
    parser.add_argument('--model', type=str, default='GraphMAE', choices=['GraphMAE', 'DGI', 'GRACE', 'GraphMAE2'])
    parser.add_argument('--method', type=str, default='sa_ot_prompt', 
                        choices=['sa_ot_prompt', 'linear_probe', 'fine_tune', 'uniprompt', 
                                 'hybrid_prompt', 'gppt', 'gpf', 'gpf_plus', 
                                 'graphprompt', 'all_in_one', 'edgeprompt', 'edgeprompt_plus'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--trails', type=int, default=30)
    parser.add_argument('--down_lr', type=float, default=0.005)
    parser.add_argument('--clf_lr', type=float, default=0.05)
    parser.add_argument('--down_wd', type=float, default=5e-5)
    parser.add_argument('--down_epochs', type=int, default=500)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--num_prompts', type=int, default=10)
    parser.add_argument('--ot_epsilon', type=float, default=0.1)
    parser.add_argument('--ot_beta', type=float, default=0.01)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--feat_mask', type=float, default=0.0)
    
    args = parser.parse_args()
    set_seed(args.seed)
    run(args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))