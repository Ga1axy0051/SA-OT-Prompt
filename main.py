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

def compute_homophily(edge_index, y):
    """ 计算图的同配性分数 (Edge Homophily) """
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
    
    # ==========================================
    # 1. 加载与预处理数据
    # ==========================================
    data, input_dim, output_dim = load_dataset(args.dataset, data_dir="./data/raw")
    data = data.to(device)
    
    # 🛡️ 扰动模块 A：特征掩码 (Feature Masking) - 模拟特征缺失
    if args.feat_mask > 0:
        print(f">>> Applying {args.feat_mask*100}% Feature Masking (Zeroing out features)...")
        # 生成掩码：保留 (1 - feat_mask) 的特征，丢弃 feat_mask 的特征
        mask = torch.rand_like(data.x) > args.feat_mask
        data.x = data.x * mask

    # ☠️ 扰动模块 B：结构噪声 (Edge Noise) - 模拟错连边
    if args.noise > 0:
        print(f">>> Injecting {args.noise*100}% heterophilic noise edges...")
        data.edge_index = inject_noise_edges(data.edge_index, data.y, args.noise)
    
    # 边权重归一化
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32, device=device)
    edge_index, edge_weight = add_self_loops(data.edge_index, edge_weight)
    edge_weight = normalize_edge(edge_index, edge_weight, data.num_nodes).to(device)
    edge_index = edge_index.to(device)

    # ==========================================
    # 2. Stage 1: 加载/训练预训练基座
    # ==========================================
    print(">>> Stage 1: Loading/Training Base Encoder...")
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

    # ==========================================
    # 3. Stage 2: 下游 Few-shot 适配
    # ==========================================
    print(">>> Stage 2: Few-shot Downstream Adaptation...")
    test_accs, f1s, rocs = [], [], []
    down_loss_fn = nn.CrossEntropyLoss()

    for trail in range(1, args.trails + 1):
        print(f"\n--- Trail {trail}/{args.trails} ---")
        current_seed = args.seed + trail 
        data = generate_few_shot_splits(data, output_dim, shot=args.shot, seed=current_seed)
        
        classifier = LogReg(args.hid_dim, output_dim).to(device)
        prompt = None # 提前占位，防止 UnboundLocalError
        
        # 初始化 Prompt 或 调整基座梯度
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

        with tqdm(total=args.down_epochs, desc=f'Adapt Trail {trail}') as pbar:
            for epoch in range(args.down_epochs):
                if prompt is not None: prompt.train()
                classifier.train()
                optimizer_down.zero_grad()
                
                # --- 前向传播 ---
                if args.method in ['sa_ot_prompt', 'hybrid_prompt']:
                    x_ad, ot_l, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
                    c_idx, c_w = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                    embeds = base_encoder.embed(x_ad, c_idx, c_w)
                    ot_loss_val = ot_l
                elif args.method == 'uniprompt':
                    pt_idx, pt_w = prompt()
                    comb_index, comb_weight = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                    embeds = base_encoder.embed(data.x, comb_index, comb_weight)
                else:
                    embeds = base_encoder.embed(data.x, edge_index, edge_weight)
                    ot_loss_val = torch.tensor(0.0).to(device)
                
                logits = classifier(embeds)
                train_loss = down_loss_fn(logits[data.train_mask], data.y[data.train_mask])
                
                loss = train_loss + (args.ot_beta * ot_loss_val if args.method == 'sa_ot_prompt' else 0)
                loss.backward()
                optimizer_down.step()

                # --- 验证逻辑 ---
                with torch.no_grad():
                    val_logits = classifier(embeds)
                    v_loss = down_loss_fn(val_logits[data.val_mask], data.y[data.val_mask])
                    v_acc = (val_logits[data.val_mask].argmax(1) == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

                pbar.set_postfix({'val_acc': f"{v_acc:.4f}", 'train_loss': f"{train_loss.item():.4f}"})
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

        # ==========================================
        # 4. 测试评估
        # ==========================================
        classifier.load_state_dict({k: v.to(device) for k, v in best_classifier_state.items()})
        classifier.eval()
        with torch.no_grad():
            if args.method in ['sa_ot_prompt', 'hybrid_prompt']:
                prompt.load_state_dict({k: v.to(device) for k, v in best_prompt_state.items()})
                prompt.eval()
                x_ad, _, pt_idx, pt_w = prompt(data.x, edge_index, edge_weight)
                comb_index, comb_weight = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(x_ad, comb_index, comb_weight)
                h_orig = compute_homophily(data.edge_index, data.y)
                h_new = compute_homophily(comb_index, data.y)
                print(f"   [Structural Analysis] Original Homophily: {h_orig:.4f} -> Prompted: {h_new:.4f}")
            
            elif args.method == 'uniprompt':
                prompt.load_state_dict({k: v.to(device) for k, v in best_prompt_state.items()})
                prompt.eval()
                pt_idx, pt_w = prompt()
                comb_index, comb_weight = prompt.edge_fuse(edge_index, edge_weight, pt_idx, pt_w, args.tau)
                embeds = base_encoder.embed(data.x, comb_index, comb_weight)
                h_orig = compute_homophily(data.edge_index, data.y)
                h_new = compute_homophily(comb_index, data.y)
                print(f"   [Structural Analysis] Original Homophily: {h_orig:.4f} -> Prompted: {h_new:.4f}")
            
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

            # 显存清道夫
            del classifier, optimizer_down
            if prompt is not None:
                del prompt
            torch.cuda.empty_cache()

    # ==========================================
    # 5. 最终汇总
    # ==========================================
    print("\n" + "="*50)
    print(f"Final Report: {args.dataset} | Method: {args.method}")
    if len(test_accs) > 0:
        print(f"Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        print(f"Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        if len(rocs) > 0:
            print(f"AUROC   : {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--model', type=str, default='GraphMAE')
    parser.add_argument('--method', type=str, default='sa_ot_prompt', choices=['sa_ot_prompt', 'linear_probe', 'fine_tune', 'uniprompt', 'hybrid_prompt'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
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
    parser.add_argument('--alpha', type=float, default=0.5, help='Fusion weight for HybridPrompt')
    
    # 🛡️ 核心新增：控制特征噪声与结构噪声的接口
    parser.add_argument('--noise', type=float, default=0.0, help='Ratio of heterophilic noise edges')
    parser.add_argument('--feat_mask', type=float, default=0.0, help='Ratio of feature dimensions to randomly mask (0.0 to 1.0)')
    
    args = parser.parse_args()
    set_seed(args.seed)
    run(args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))