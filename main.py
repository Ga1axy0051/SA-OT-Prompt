import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import importlib
from tqdm import tqdm
from torch_geometric.utils import add_self_loops, dropout_adj
from torch_geometric.nn import GCNConv
import dgl  

# 导入自定义模块
from utils.data_loader import load_dataset, generate_few_shot_splits, inject_noise_edges
from models.ot_prompt import SAOTPrompt
from models.uniprompt import UniPrompt
from models.hybrid_prompt import HybridPrompt
from models.daprompt import DAPrompt_Prompt
from models.hsgppt import HSGPPT_Prompt
from models.pronog import ProNoG_Prompt
from utils.legacy_utils import normalize_edge, NodeEva

# 导入基座模型
from models.graphmae import build_model
from models.DGI import DGI, DGI_process
from models.GRACE import Encoder, Model as GRACE_Model, drop_feature
from models.Base import LogReg
sys.path.append(os.path.abspath('./pretrain_model'))

# MaskGAE 适配模块
from maskgae.model import MaskGAE, GNNEncoder, DotEdgeDecoder, DegreeDecoder, EdgeDecoder

# 导入所有 Baseline 的 Prompt 模块
from models import GPPT_Prompt, GPF_Prompt, GPF_plus_Prompt, EdgePrompt, EdgePrompt_plus, GraphPrompt_Prompt, AllInOne_Prompt

# ==============================================================
# 📊 动力学探针函数 (用于量化结构与表征的坍塌)
# ==============================================================
def compute_homophily(edge_index, y):
    src, dst = edge_index
    same_label = (y[src] == y[dst]).sum().item()
    return same_label / edge_index.size(1) if edge_index.size(1) > 0 else 0

def compute_degree_gini(edge_index, edge_weight, num_nodes):
    """
    计算图拓扑的度基尼系数 (Degree Gini Coefficient)
    接近 1 说明连边极度集中（模式坍塌至星型图），接近 0 说明度分布均匀
    """
    device = edge_index.device
    # 🟢 动态计算最大节点数，防止带 Prompt 的拓扑引发 scatter_add 越界
    actual_num_nodes = max(num_nodes, int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else num_nodes)
    
    degree = torch.zeros(actual_num_nodes, device=device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device)
    
    degree.scatter_add_(0, edge_index[1], edge_weight)
    
    degree_sorted, _ = torch.sort(degree)
    n = degree_sorted.size(0)
    index = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    gini = ((2 * index - n - 1) * degree_sorted).sum() / (n * degree_sorted.sum() + 1e-8)
    return gini.item()

def compute_dirichlet_energy(embeds, edge_index, edge_weight):
    """
    计算特征的狄利克雷能量 (Dirichlet Energy)
    能量趋近于 0 说明全图特征过度平滑，丧失类别可分性
    """
    # 🟢 过滤掉超出 embeds 索引的虚拟 Prompt 连边，只专注原图节点的特征平滑度监控
    valid_mask = (edge_index[0] < embeds.size(0)) & (edge_index[1] < embeds.size(0))
    src = edge_index[0][valid_mask]
    dst = edge_index[1][valid_mask]
    
    if edge_weight is None:
        ew = torch.ones(src.size(0), device=embeds.device)
    else:
        ew = edge_weight[valid_mask]
    
    if src.size(0) == 0:
        return 0.0
        
    diff = embeds[src] - embeds[dst]
    energy = (ew * diff.pow(2).sum(dim=-1)).sum() / embeds.size(0)
    return energy.item()


# ==============================================================
# ⚙️ 核心流程引擎
# ==============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def run(args, device):
    # ---------------- 1. 数据加载与鲁棒性校验 ----------------
    data, input_dim, output_dim = load_dataset(args.dataset, data_dir="./data/raw")
    data = data.to(device)
    
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

    # ---------------- 2. 预训练基座加载与冻结 ----------------
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
        print(f"=== 正在加载 GraphMAE2 ({args.dataset}) ===")
        graphmae2_root = os.path.abspath('./pretrain_model/graphmae2')
        curr_dir = os.getcwd()
        old_sys_path = sys.path[:]
        modules_backup = {k: v for k, v in sys.modules.items() if k == 'models' or k.startswith('models.') or k == 'utils' or k.startswith('utils.')}
        for k in modules_backup: sys.modules.pop(k)
        
        new_path = [graphmae2_root] + [p for p in old_sys_path if os.path.abspath(p) != os.path.abspath(curr_dir) and p != '']
        sys.path = new_path
        
        try:
            import models as mae2_pkg
            importlib.reload(mae2_pkg)
            build_mae2 = mae2_pkg.build_model
            
            cfg = {
                'lr': args.lr, 'weight_decay': args.wd, 'optimizer': 'adam',
                'num_hidden': args.hid_dim, 'num_heads': 4, 'concat_out': False,
                'num_out_heads': 1, 'num_layers': 2, 'num_dec_layers': 1, 
                'mask_rate': 0.5, 'mask_method': 'random', 'remask_rate': 0.5, 
                'remask_method': 'random', 'num_remasking': 3, 'replace_rate': 0.05, 
                'drop_edge_rate': 0.0, 'encoder': 'gat', 'decoder': 'gat', 
                'activation': 'prelu', 'negative_slope': 0.2, 'attn_drop': 0.1, 
                'in_drop': 0.2, 'norm': 'layernorm', 'residual': True, 
                'loss_fn': 'sce', 'alpha_l': 2.0, 'lam': 1.0, 'delayed_ema_epoch': 0, 
                'momentum': 0.996, 'type_grad': 'grad', 'pooling': 'mean', 
                'zero_init': False, 'is_sparse': False, 'num_features': input_dim, 'dataset': args.dataset
            }
            mae2_args = Namespace(**cfg)
            raw_model = build_mae2(mae2_args).to(device)
        finally:
            for k in list(sys.modules.keys()):
                if k == 'models' or k.startswith('models.'): sys.modules.pop(k)
            sys.modules.update(modules_backup)
            sys.path = old_sys_path

        ckpt_dir = os.path.join(graphmae2_root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        mae2_ckpt_path = os.path.join(ckpt_dir, f"{args.dataset}.pt")
        
        try:
            raw_model.load_state_dict(torch.load(mae2_ckpt_path, map_location=device))
            print(f">> 成功挂载权重: {args.dataset}.pt")
        except:
            print(f">> 🚀 权重缺失，启动 GraphMAE2 现场即时预训练...")
            raw_model.train()
            optimizer_pt = torch.optim.Adam(raw_model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            g_pt = dgl.add_self_loop(dgl.remove_self_loop(dgl.graph((edge_index[0], edge_index[1]), num_nodes=data.num_nodes).to(device)))
            for pt_epoch in range(args.epochs): 
                loss_pt = raw_model(g_pt, data.x)
                optimizer_pt.zero_grad()
                loss_pt.backward()
                optimizer_pt.step()
                if (pt_epoch + 1) % 100 == 0: print(f"   - Base Epoch {pt_epoch+1}/{args.epochs} | Loss: {loss_pt.item():.4f}")
            torch.save(raw_model.state_dict(), mae2_ckpt_path)
            model_save_path = mae2_ckpt_path 

        raw_model.eval()
        for param in raw_model.encoder.parameters(): param.requires_grad = False

    elif args.model == 'MaskGAE':
        print(f"=== 正在加载 MaskGAE ({args.dataset}) ===")
        encoder = GNNEncoder(in_channels=input_dim, hidden_channels=args.hid_dim, out_channels=args.hid_dim, num_layers=1, dropout=0.8, bn=False, layer="gcn", activation="elu")
        maskgae_model = MaskGAE(encoder, EdgeDecoder(args.hid_dim, 64, num_layers=2, dropout=0.2), DegreeDecoder(args.hid_dim, 64, num_layers=2, dropout=0.2), mask=None).to(device)
        ckpt_path = f"./pretrain_model/maskgae/pretrain_weights/maskgae_{args.dataset.lower()}.pt"
        maskgae_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f">> 成功挂载 MaskGAE 权重: {ckpt_path}")
        raw_model = maskgae_model.encoder
        raw_model.eval()
        for param in raw_model.parameters(): param.requires_grad = False

    # 老模型自动预训练
    if not os.path.exists(model_save_path) and args.model not in ['GraphMAE2', 'MaskGAE']:
        print(f"🔥 未找到 {args.model} 权重，正在全自动执行预训练...")
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=args.lr, weight_decay=args.wd)
        if args.model == 'DGI': loss_func = nn.BCEWithLogitsLoss()
        raw_model.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            if args.model == 'GraphMAE': loss, _ = raw_model(data.x, edge_index, edge_weight)
            elif args.model == 'DGI':
                shuf_x, lbl = DGI_process(data.num_nodes, data.x)
                loss = loss_func(raw_model(data.x, shuf_x.to(device), edge_index, edge_weight, None, None, None), lbl.to(device))
            elif args.model == 'GRACE':
                edge_index_1, edge_weight_1 = dropout_adj(edge_index, edge_weight, p=0.2)
                edge_index_2, edge_weight_2 = dropout_adj(edge_index, edge_weight, p=0.2)
                loss = raw_model.loss(raw_model(drop_feature(data.x, 0.2), edge_index_1, edge_weight_1), raw_model(drop_feature(data.x, 0.2), edge_index_2, edge_weight_2), batch_size=0)
            loss.backward()
            optimizer.step()
        torch.save(raw_model.state_dict(), model_save_path)
    
    if args.model not in ['GraphMAE2', 'MaskGAE']:
        raw_model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

    # 透明代理层
    class BackboneWrapper:
        def __init__(self, model_type, model):
            self.model_type, self.model = model_type, model
        def embed(self, x, edge_index, edge_weight=None):
            if self.model_type == 'DGI': return self.model.embed(x, edge_index, edge_weight, None)[0]
            elif self.model_type == 'GraphMAE2':
                g_eval = dgl.add_self_loop(dgl.remove_self_loop(dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0]).to(x.device)))
                return self.model.embed(g_eval, x)
            elif self.model_type == 'MaskGAE': return self.model.get_embedding(x, edge_index)
            return self.model.embed(x, edge_index, edge_weight)
        def parameters(self): return self.model.encoder.parameters() if self.model_type == 'GraphMAE2' else self.model.parameters()
        def state_dict(self): return self.model.encoder.state_dict() if self.model_type == 'GraphMAE2' else self.model.state_dict()
        def load_state_dict(self, sd): self.model.encoder.load_state_dict(sd) if self.model_type == 'GraphMAE2' else self.model.load_state_dict(sd)
        def __getattr__(self, name): return getattr(self.model, name)

    base_encoder = BackboneWrapper(args.model, raw_model)

    # ---------------- 3. 下游 Few-shot 适配与动力学追踪 ----------------
    test_accs = []
    down_loss_fn = nn.CrossEntropyLoss()
    all_dynamics_records = []  # 🟢 记录所有 trail 的动力学数据

    for trail in range(1, args.trails + 1):
        current_seed = args.seed + trail 
        data = generate_few_shot_splits(data, output_dim, shot=args.shot, seed=current_seed)
        classifier = LogReg(args.hid_dim, output_dim).to(device)
        prompt = None 
        
        # 🟢 单次跑的动力学记录仪
        dynamics_records = {'epoch': [], 'gini': [], 'energy': [], 'val_acc': []}
        
        # 初始化 Prompt
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

        elif args.method == 'daprompt':
            prompt = DAPrompt_Prompt(in_dim=input_dim, num_classes=output_dim, 
                                     num_structs=2, outer_thre=0.2, device=device).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": prompt.parameters(), "lr": args.down_lr}, 
                                               {"params": classifier.parameters(), "lr": args.clf_lr}], 
                                               weight_decay=args.down_wd)
        
        elif args.method == 'hsgppt':
            prompt = HSGPPT_Prompt(in_dim=input_dim, num_nodes_prompt=10, 
                                   tau_inner=0.2, tau_cross=0.4, device=device).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": prompt.parameters(), "lr": args.down_lr}, 
                                               {"params": classifier.parameters(), "lr": args.clf_lr}], 
                                               weight_decay=args.down_wd)
            
        elif args.method == 'pronog':
            prompt = ProNoG_Prompt(in_dim=input_dim, hidden_dim=args.hid_dim, device=device).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": prompt.parameters(), "lr": args.down_lr}, 
                                               {"params": classifier.parameters(), "lr": args.clf_lr}], 
                                               weight_decay=args.down_wd)

        elif args.method == 'hybrid_prompt':
            prompt = HybridPrompt(data.x, input_dim, args.num_prompts, args.ot_epsilon, args.k, args.alpha).to(device)
            base_encoder.eval(); [p.requires_grad_(False) for p in base_encoder.parameters()]
            optimizer_down = torch.optim.Adam([{"params": prompt.parameters(), "lr": args.down_lr}, {"params": classifier.parameters(), "lr": args.clf_lr}], weight_decay=args.down_wd)

        best_val_acc = -1.0
        cnt_wait = 0
        best_classifier_state, best_prompt_state, best_base_state = None, None, None

        for epoch in range(args.down_epochs):
            if prompt is not None: prompt.train()
            classifier.train()
            optimizer_down.zero_grad()
            ot_loss_val = torch.tensor(0.0).to(device)
            
            # 🔍 默认拓扑（用于探针拦截）
            actual_edge_index = edge_index
            actual_edge_weight = edge_weight

            if args.method in ['sa_ot_prompt', 'hybrid_prompt']:
                x_ad, ot_l, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
                c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(x_ad, c_idx, c_w)
                ot_loss_val = ot_l
                actual_edge_index, actual_edge_weight = c_idx, c_w
                
            elif args.method == 'uniprompt':
                pt_idx, pt_w = prompt()
                c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(data.x, c_idx, c_w)
                actual_edge_index, actual_edge_weight = c_idx, c_w
                
            elif args.method in ['gpf', 'gpf_plus']:
                embeds = base_encoder.embed(prompt(data.x), edge_index, edge_weight)
                
            elif args.method == 'all_in_one':
                new_x, ni, nw = prompt(data.x, edge_index, edge_weight)
                embeds = base_encoder.embed(new_x, ni, nw)[:data.num_nodes]
                actual_edge_index, actual_edge_weight = ni, nw
                
            elif args.method in ['edgeprompt', 'edgeprompt_plus']:
                if hasattr(prompt, 'get_rewired_graph'):
                    new_x, mod_idx, mod_w = prompt.get_rewired_graph(data.x, edge_index)
                    embeds = base_encoder.embed(new_x, mod_idx, mod_w)
                    actual_edge_index, actual_edge_weight = mod_idx, mod_w
                else:
                    embeds = base_encoder.embed(prompt(data.x, edge_index), edge_index, edge_weight)
                    
            elif args.method == 'graphprompt':
                embeds = prompt(base_encoder.embed(data.x, edge_index, edge_weight))

            elif args.method == 'daprompt':
                new_x, mod_idx, mod_w = prompt(data.x, edge_index)
                embeds_all = base_encoder.embed(new_x, mod_idx, mod_w)
                embeds = embeds_all[:data.num_nodes]
                actual_edge_index, actual_edge_weight = mod_idx, mod_w
            
            elif args.method == 'hsgppt':
                new_x, mod_idx, mod_w = prompt(data.x, edge_index)
                embeds_all = base_encoder.embed(new_x, mod_idx, mod_w)
                embeds = embeds_all[:data.num_nodes]
                actual_edge_index, actual_edge_weight = mod_idx, mod_w

            elif args.method == 'pronog':
                new_x, mod_idx, mod_w = prompt(data.x, edge_index)
                embeds = base_encoder.embed(new_x, mod_idx, mod_w)
                actual_edge_index, actual_edge_weight = mod_idx, mod_w
                
            else:
                embeds = base_encoder.embed(data.x, edge_index, edge_weight)
            
            logits = classifier(embeds)
            loss = down_loss_fn(logits[data.train_mask], data.y[data.train_mask]) + (args.ot_beta * ot_loss_val if args.method == 'sa_ot_prompt' else 0)
            loss.backward()
            optimizer_down.step()

            # ---------------- 验证逻辑 ----------------
            with torch.no_grad():
                classifier.eval()
                v_logits = classifier(embeds)
                v_acc = (v_logits[data.val_mask].argmax(1) == data.y[data.val_mask]).sum().item() / max(data.val_mask.sum().item(), 1)
                
                if v_acc > best_val_acc:
                    best_val_acc, cnt_wait = v_acc, 0
                    best_classifier_state = {k: v.cpu() for k, v in classifier.state_dict().items()}
                    if prompt is not None: best_prompt_state = {k: v.cpu() for k, v in prompt.state_dict().items()}
                    if args.method == 'fine_tune': best_base_state = {k: v.cpu() for k, v in base_encoder.state_dict().items()}
                else:
                    cnt_wait += 1
                    if not args.track_dynamics and cnt_wait >= args.patience: break
            
            # 🟢 [探针监控核心] 记录坍塌动力学指标
            if args.track_dynamics and epoch % 10 == 0:
                with torch.no_grad():
                    current_gini = compute_degree_gini(actual_edge_index, actual_edge_weight, data.num_nodes)
                    current_energy = compute_dirichlet_energy(embeds, actual_edge_index, actual_edge_weight)
                    
                    dynamics_records['epoch'].append(epoch)
                    dynamics_records['gini'].append(current_gini)
                    dynamics_records['energy'].append(current_energy)
                    dynamics_records['val_acc'].append(v_acc)

        # ---------------- 收集单次实验动力学记录 ----------------
        if args.track_dynamics:
            all_dynamics_records.append(dynamics_records)

        # ---------------- 测试评估 ----------------
        classifier.load_state_dict({k: v.to(device) for k, v in best_classifier_state.items()})
        classifier.eval()
        if prompt is not None: 
            prompt.load_state_dict({k: v.to(device) for k, v in best_prompt_state.items()})
            prompt.eval()
            
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
                if hasattr(prompt, 'get_rewired_graph'):
                    nx, mod_idx, mod_w = prompt.get_rewired_graph(data.x, edge_index)
                    embeds = base_encoder.embed(nx, mod_idx, mod_w)
                else:
                    embeds = base_encoder.embed(prompt(data.x, edge_index), edge_index, edge_weight)
            elif args.method == 'graphprompt':
                embeds = prompt(base_encoder.embed(data.x, edge_index, edge_weight))
            elif args.method == 'daprompt':
                new_x, mod_idx, mod_w = prompt(data.x, edge_index)
                embeds = base_encoder.embed(new_x, mod_idx, mod_w)[:data.num_nodes]
            elif args.method == 'hsgppt':
                new_x, mod_idx, mod_w = prompt(data.x, edge_index)
                embeds_all = base_encoder.embed(new_x, mod_idx, mod_w)
                embeds = embeds_all[:data.num_nodes]
                actual_edge_index, actual_edge_weight = mod_idx, mod_w
            elif args.method == 'pronog':
                new_x, mod_idx, mod_w = prompt(data.x, edge_index)
                embeds = base_encoder.embed(new_x, mod_idx, mod_w)
                actual_edge_index, actual_edge_weight = mod_idx, mod_w
            else:
                if args.method == 'fine_tune': base_encoder.load_state_dict({k: v.to(device) for k, v in best_base_state.items()})
                embeds = base_encoder.embed(data.x, edge_index, edge_weight)
            
            t_acc, _, _, _ = NodeEva(classifier(embeds), torch.nonzero(data.test_mask).squeeze(), data, output_dim, device)
            test_accs.append(t_acc)
        torch.cuda.empty_cache()

    print(f"[{args.model}] + [{args.method}] Final Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

    # ==============================================================
    # 📈 动力学落盘机制 (所有实验结束后，基于均值保存)
    # ==============================================================
    if args.track_dynamics and len(all_dynamics_records) > 0:
        avg_records = {'epoch': all_dynamics_records[0]['epoch']}
        avg_records['gini'] = np.mean([d['gini'] for d in all_dynamics_records], axis=0).tolist()
        avg_records['energy'] = np.mean([d['energy'] for d in all_dynamics_records], axis=0).tolist()
        avg_records['val_acc'] = np.mean([d['val_acc'] for d in all_dynamics_records], axis=0).tolist()
        
        record_file = args.dynamics_log_path if args.dynamics_log_path else f"{args.method}_{args.dataset}_{args.shot}shot_dynamics.json"
        with open(record_file, 'w') as f:
            json.dump(avg_records, f)
        print(f"📊 严谨动力学演化数据 (基于 {args.trails} 次实验取均值) 已保存至: {record_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--model', type=str, default='GraphMAE', choices=['GraphMAE', 'DGI', 'GRACE', 'GraphMAE2', 'MaskGAE']) 
    parser.add_argument('--method', type=str, default='sa_ot_prompt', 
                        choices=['sa_ot_prompt', 'linear_probe', 'fine_tune', 'uniprompt', 
                                 'hybrid_prompt', 'gppt', 'gpf', 'gpf_plus', 
                                 'graphprompt', 'all_in_one', 'edgeprompt', 'edgeprompt_plus', 'daprompt', 'hsgppt', 'pronog'])
    parser.add_argument('--seed', type=int, default=42)
    
    # 🔍 动力学探针机制开关
    parser.add_argument('--track_dynamics', action='store_true', help="开启以捕获拓扑与特征的坍塌动力学")
    parser.add_argument('--dynamics_log_path', type=str, default='', help="动力学追踪结果保存路径")

    # 🛡️ 绝对锁死区 (The Golden Baseline)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--num_prompts', type=int, default=10)
    parser.add_argument('--clf_lr', type=float, default=0.05)
    parser.add_argument('--down_epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--trails', type=int, default=30)
    parser.add_argument('--down_wd', type=float, default=5e-5)
    
    # 预训练备用参数
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=500)
    
    # 🔍 动态搜索区
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--down_lr', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--ot_beta', type=float, default=0.01)
    parser.add_argument('--ot_epsilon', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--feat_mask', type=float, default=0.0)
    
    args = parser.parse_args()
    set_seed(args.seed)
    run(args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))