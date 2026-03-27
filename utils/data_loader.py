import torch
import numpy as np
import torch_geometric.transforms as T
# 👑 新增了 Amazon 和 Coauthor 数据集的导入
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Amazon, Coauthor
from torch_geometric.utils import negative_sampling

def load_dataset(dataset_name, data_dir="./data/raw"):
    """
    统一加载标准图数据集
    """
    dataset_name = dataset_name.lower()
    
    # 特征归一化：保证 OT 算出来的 Cost 距离数值稳定，防崩溃神器
    transform = T.NormalizeFeatures() 
    
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=data_dir, name=dataset_name.capitalize(), transform=transform)
        
    elif dataset_name in ['texas', 'wisconsin', 'cornell']:
        dataset = WebKB(root=data_dir, name=dataset_name.capitalize(), transform=transform)
        
    elif dataset_name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=data_dir, name=dataset_name.capitalize(), transform=transform)

    elif dataset_name == 'actor':
        from torch_geometric.datasets import Actor
        import os
        dataset = Actor(root=os.path.join(data_dir, 'Actor'))
        
    # 👑 杀手级主场 1：Amazon 商品购买网络 (特征极弱，高度依赖购买结构)
    elif dataset_name in ['amazon-computers', 'amazon-photo']:
        name = 'Computers' if 'computers' in dataset_name else 'Photo'
        dataset = Amazon(root=data_dir, name=name, transform=transform)
        
    # 👑 杀手级主场 2：Coauthor 学术合作网络 (高阶社区密集，OT 的完美猎物)
    elif dataset_name in ['coauthor-cs', 'coauthor-physics']:
        name = 'CS' if 'cs' in dataset_name else 'Physics'
        dataset = Coauthor(root=data_dir, name=name, transform=transform)
        
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported yet!")
        
    data = dataset[0]
    return data, dataset.num_features, dataset.num_classes


def generate_few_shot_splits(data, num_classes, shot=1, seed=42):
    """
    动态生成 Few-shot 掩码 (Masks)。
    自带极度长尾类别保护机制，绝对公平，杜绝作弊。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    labels = data.y.cpu().numpy()
    
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        
        # 极端长尾保护机制
        if len(idx_c) == 0:
            continue
            
        elif len(idx_c) <= shot:
            # 样本极少，全给训练集保命
            train_idx = idx_c
            val_idx = []
            test_idx = []
        else:
            # 正常情况切分
            train_idx = idx_c[:shot]
            remaining = len(idx_c) - shot
            val_size = min(15, remaining // 2)
            val_size = max(1, val_size) if remaining > 0 else 0
            
            val_idx = idx_c[shot : shot + val_size]
            test_idx = idx_c[shot + val_size :]
            
        train_mask[train_idx] = True
        if len(val_idx) > 0:
            val_mask[val_idx] = True
        if len(test_idx) > 0:
            test_mask[test_idx] = True
            
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print(f"Data Split [Seed: {seed} | Shot: {shot}]: "
          f"Train: {int(train_mask.sum())} | "
          f"Val: {int(val_mask.sum())} | "
          f"Test: {int(test_mask.sum())}")
          
    return data

def inject_noise_edges(edge_index, y, noise_ratio=0.2):
    """ 👑 绝不卡顿的向量化投毒函数 """
    if noise_ratio <= 0: return edge_index
    
    device = edge_index.device
    num_nodes = y.size(0)
    num_edges = edge_index.size(1)
    num_noise = int(num_edges * noise_ratio)

    all_noise_src, all_noise_dst = [], []
    found = 0
    # 批量采样，直到凑够异配边
    while found < num_noise:
        src = torch.randint(0, num_nodes, (num_noise * 2,), device=device)
        dst = torch.randint(0, num_nodes, (num_noise * 2,), device=device)
        mask = (y[src] != y[dst]) & (src != dst) # 异配且非自环
        all_noise_src.append(src[mask])
        all_noise_dst.append(dst[mask])
        found += mask.sum().item()
    
    noise_src = torch.cat(all_noise_src)[:num_noise]
    noise_dst = torch.cat(all_noise_dst)[:num_noise]
    noise_edge_index = torch.stack([noise_src, noise_dst], dim=0)
    return torch.cat([edge_index, noise_edge_index], dim=1)

# ==========================================
# 测试入口
# ==========================================
if __name__ == "__main__":
    print("Testing DataLoader with New Datasets...")
    # 测试咱们的新主场
    data, input_dim, num_classes = load_dataset("amazon-photo", data_dir="./data/raw")
    print(f"Loaded Amazon-Photo: Nodes={data.num_nodes}, Edges={data.num_edges}, Features={input_dim}, Classes={num_classes}")
    
    data = generate_few_shot_splits(data, num_classes, shot=1, seed=42)
    print("DataLoader test passed perfectly! 🚀")