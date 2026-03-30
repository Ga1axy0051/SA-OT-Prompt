import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GCNConv  # 🟢 [新增]: 用于 GRACE 编码器的实例化

# 导入自定义模块
from utils.data_loader import load_dataset, generate_few_shot_splits, inject_noise_edges
from models.ot_prompt import SAOTPrompt
from models.uniprompt import UniPrompt
from models.hybrid_prompt import HybridPrompt
from utils.legacy_utils import normalize_edge, NodeEva

# 导入基座模型
from models.graphmae import build_model
from models.DGI import DGI
from models.GRACE import Encoder, Model as GRACE_Model
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
    # 1. 加载与预处理数据
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
    # 2. Stage 1: 加载/训练预训练基座 (🟢 核心修改区域)
    # ==============================================================
    model_save_path = f'./pretrain_model/{args.model}/{args.dataset}_hid{args.hid_dim}.pkl'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # [1] 根据参数实例化三大不同的预训练底座
    if args.model == 'GraphMAE':
        raw_model = build_model(num_hidden=args.hid_dim, num_features=input_dim).to(device)
    elif args.model == 'DGI':
        raw_model = DGI(input_dim, args.hid_dim, 'prelu').to(device)
    elif args.model == 'GRACE':
        encoder = Encoder(input_dim, args.hid_dim, nn.PReLU(), base_model=GCNConv, k=2).to(device)
        raw_model = GRACE_Model(encoder, args.hid_dim, args.hid_dim, 0.5).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # [2] 权重加载与预训练拦截
    if not os.path.exists(model_save_path):
        if args.model != 'GraphMAE':
            # 强行拦截！如果是 DGI/GRACE 但没找到权重，提示必须用 UniPrompt 官方脚本先跑
            raise FileNotFoundError(f"⚠️ 找不到 {args.model} 的权重 ({model_save_path})。请先使用 UniPrompt 的脚本完成预训练！")
        
        print("未找到 GraphMAE 权重，正在自动执行预训练...")
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=args.lr, weight_decay=args.wd)
        raw_model.train()
        for _ in range(args.epochs):
            optimizer.zero_grad()
            loss, _ = raw_model(data.x, edge_index, edge_weight)
            loss.backward()
            optimizer.step()
        torch.save(raw_model.state_dict(), model_save_path)
    
    # 统一加载权重
    raw_model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

    # [3] 透明代理层 (BackboneWrapper)：抹平 DGI 返回值的差异
    class BackboneWrapper:
        def __init__(self, model_type, model):
            self.model_type = model_type
            self.model = model
            
        def embed(self, x, edge_index, edge_weight):
            if self.model_type == 'DGI':
                # DGI 的 embed 需要一个 None 掩码，且返回值为 (z, c)
                z, _ = self.model.embed(x, edge_index, edge_weight, None)
                return z
            else:
                return self.model.embed(x, edge_index, edge_weight)
                
        def __getattr__(self, name):
            # 任何非 embed 的调用 (比如 .parameters(), .eval(), .load_state_dict())
            # 都会被无缝原样传递给内部真实的 raw_model
            return getattr(self.model, name)

    # 用代理层将原始模型包起来！下游代码从此高枕无忧！
    base_encoder = BackboneWrapper(args.model, raw_model)

    # ==============================================================
    # 3. Stage 2: 下游 Few-shot 适配 (下游代码完全无需任何修改！)
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
                {"params": classifier.parameters(), "lr": 0.05}
            ], weight_decay=args.down_wd)
            
        elif args.method == 'linear_probe':
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam(classifier.parameters(), lr=0.05, weight_decay=args.down_wd)
            
        elif args.method == 'fine_tune':
            base_encoder.train()
            for param in base_encoder.parameters(): param.requires_grad = True
            optimizer_down = torch.optim.Adam([
                {"params": base_encoder.parameters(), "lr": 0.0001},
                {"params": classifier.parameters(), "lr": 0.05}
            ], weight_decay=args.down_wd)
        
        elif args.method == 'uniprompt':
            prompt = UniPrompt(x=data.x, k=args.k, metric='cosine', alpha=1.0, num_nodes=data.num_nodes).to(device)
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": 0.001, "weight_decay": 5e-4},
                {"params": classifier.parameters(), "lr": 0.05}
            ])

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
                {"params": classifier.parameters(), "lr": 0.05}
            ], weight_decay=args.down_wd)

        elif args.method in ['edgeprompt', 'edgeprompt_plus']:
            if args.method == 'edgeprompt': prompt = EdgePrompt(in_dim=input_dim).to(device)
            else: prompt = EdgePrompt_plus(in_dim=input_dim).to(device)
            
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": 0.05}
            ], weight_decay=args.down_wd)

        elif args.method == 'graphprompt':
            prompt = GraphPrompt_Prompt(in_dim=args.hid_dim).to(device)
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": 0.05}
            ], weight_decay=args.down_wd)
        
        elif args.method == 'hybrid_prompt':
            prompt = HybridPrompt(data.x, input_dim, args.num_prompts, args.ot_epsilon, args.k, args.alpha).to(device)
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": 0.05}
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
    # 🟢 增加 choices 安全限制，明确三大底座
    parser.add_argument('--model', type=str, default='GraphMAE', choices=['GraphMAE', 'DGI', 'GRACE'])
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