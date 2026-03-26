import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import add_self_loops

# 导入自定义模块
from utils.data_loader import load_dataset, generate_few_shot_splits, inject_noise_edges
from models.ot_prompt import SAOTPrompt
from models.uniprompt import UniPrompt
from models.hybrid_prompt import HybridPrompt
from utils.legacy_utils import normalize_edge, NodeEva

# 导入基座模型
from models.graphmae import build_model
from models.Base import LogReg

# 🚨 导入所有 Baseline 的 Prompt 模块
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
    print(f"========== Start Task: {args.dataset} | Method: {args.method} ==========")
    
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

    # 2. Stage 1: 加载/训练预训练基座
    model_save_path = f'./pretrain_model/{args.model}/{args.dataset}_hid{args.hid_dim}.pkl'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    base_encoder = build_model(num_hidden=args.hid_dim, num_features=input_dim).to(device)
    
    if not os.path.exists(model_save_path):
        optimizer = torch.optim.Adam(base_encoder.parameters(), lr=args.lr, weight_decay=args.wd)
        base_encoder.train()
        for _ in tqdm(range(args.epochs), desc="Pretraining Base"):
            optimizer.zero_grad()
            loss, _ = base_encoder(data.x, edge_index, edge_weight)
            loss.backward()
            optimizer.step()
        torch.save(base_encoder.state_dict(), model_save_path)
    
    base_encoder.load_state_dict(torch.load(model_save_path, weights_only=True))

    # 3. Stage 2: 下游 Few-shot 适配
    test_accs, f1s, rocs = [], [], []
    down_loss_fn = nn.CrossEntropyLoss()

    for trail in range(1, args.trails + 1):
        print(f"\n--- Trail {trail}/{args.trails} ---")
        current_seed = args.seed + trail 
        data = generate_few_shot_splits(data, output_dim, shot=args.shot, seed=current_seed)
        
        classifier = LogReg(args.hid_dim, output_dim).to(device)
        prompt = None 
        
        # 🟢 【核心初始化区：分发所有方法】
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
            # Input-level prompt
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
            # Structure-level prompt
            if args.method == 'edgeprompt': prompt = EdgePrompt(in_dim=input_dim).to(device)
            else: prompt = EdgePrompt_plus(in_dim=input_dim).to(device)
            
            base_encoder.eval()
            for param in base_encoder.parameters(): param.requires_grad = False
            optimizer_down = torch.optim.Adam([
                {"params": prompt.parameters(), "lr": args.down_lr},
                {"params": classifier.parameters(), "lr": 0.05}
            ], weight_decay=args.down_wd)

        elif args.method == 'graphprompt':
            # Representation-level prompt
            prompt = GraphPrompt_Prompt(in_dim=args.hid_dim).to(device)
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

        with tqdm(total=args.down_epochs, desc=f'Adapt Trail {trail}') as pbar:
            for epoch in range(args.down_epochs):
                if prompt is not None: prompt.train()
                classifier.train()
                optimizer_down.zero_grad()
                
                # 🟢 【核心执行区：前向传播逻辑分发】
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
                    # Input-level: 修改节点特征
                    prompted_x = prompt(data.x)
                    embeds = base_encoder.embed(prompted_x, edge_index, edge_weight)
                
                elif args.method == 'all_in_one':
                    # All-in-one: 获取加入了虚拟节点的新图
                    new_x, new_edge_index, new_edge_weight = prompt(data.x, edge_index, edge_weight)
                    # 喂给基座 (此时输入的节点数变多了！)
                    full_embeds = base_encoder.embed(new_x, new_edge_index, new_edge_weight)
                    # 🚨 极其关键：裁掉尾部的 prompt 虚拟节点，只保留原图节点的 embeddings！
                    embeds = full_embeds[:data.num_nodes]
                
                elif args.method in ['edgeprompt', 'edgeprompt_plus']:
                    # Structure-level: 修改边权重
                    new_edge_weight = prompt(data.x, edge_index, edge_weight)
                    embeds = base_encoder.embed(data.x, edge_index, new_edge_weight)
                
                elif args.method == 'graphprompt':
                    # Representation-level: 修改输出表征
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

                pbar.set_postfix({'val_acc': f"{v_acc:.4f}", 'loss': f"{train_loss.item():.4f}"})
                pbar.update()

                # 早停判定
                if v_acc > best_val_acc or (v_acc == best_val_acc and v_loss.item() < best_val_loss):
                    best_val_acc = v_acc
                    best_val_loss = v_loss.item()
                    cnt_wait = 0
                    best_classifier_state = {k: v.cpu() for k, v in classifier.state_dict().items()}
                    if prompt is not None: 
                        best_prompt_state = {k: v.cpu() for k, v in prompt.state_dict().items()}
                    if args.method == 'fine_tune': 
                        best_base_state = {k: v.cpu() for k, v in base_encoder.state_dict().items()}
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
            
            elif args.method in ['gpf', 'gpf_plus', 'all_in_one']:
                prompted_x = prompt(data.x)
                embeds = base_encoder.embed(prompted_x, edge_index, edge_weight)
                
            elif args.method in ['edgeprompt', 'edgeprompt_plus']:
                new_edge_weight = prompt(data.x, edge_index, edge_weight)
                embeds = base_encoder.embed(data.x, edge_index, new_edge_weight)
                
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
            
            print(f"Trail {trail} Result: Accuracy={t_acc:.4f} | F1={t_f1:.4f} | AUROC={t_roc:.4f}")
            test_accs.append(t_acc)
            f1s.append(t_f1)
            rocs.append(t_roc)

            del classifier, optimizer_down
            if prompt is not None: del prompt
            torch.cuda.empty_cache()

    # 5. 最终汇总
    print("\n" + "="*50)
    print(f"Final Report: {args.dataset} | Method: {args.method}")
    if len(test_accs) > 0:
        print(f"Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--model', type=str, default='GraphMAE')
    
    # 🚨 这里已经把 12 种兵器全部加入了兵器谱！
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
    parser.add_argument('--trails', type=int, default=5)
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