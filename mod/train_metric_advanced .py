# src/train_metric_advanced.py (Final Fix: AMP Syntax)
"""
Advanced metric learning training script with CORRECTED CUDA/AMP/PK logic.
Features:
- Backbone: ResNet50 (default) or ViT (optional)
- Mixed loss: ArcFace (or CosFace) + Batch-hard Triplet
- PK sampling (P identities x K images)
- AMP, warmup, CosineAnnealingLR + ReduceLROnPlateau, EarlyStopping
- Auto-mapping of labels to continuous 0~num_classes-1
"""

import os
import time
import random
from collections import defaultdict
import sys 

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# *** 修正: 使用 torch.amp 導入 AMP 相關功能 ***
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from PIL import Image
from torchvision import transforms as T

# import your modules
from dataset import FolderDataset
from model import EmbeddingNet
from losses import ArcFaceLoss
from losses_cosface import CosFaceLoss

# -------------------------
# utils
# -------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

train_transform = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)), 
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.25,0.25,0.25,0.1), 
    T.RandomRotation(15), 
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def pil_loader(path):
    with Image.open(path) as img:
        return img.convert('RGB')

# -------------------------
# pairwise / triplet utils
# -------------------------
def pairwise_distance(embeddings, squared=False):
    norm_sq = torch.sum(embeddings**2, dim=1, keepdim=True)
    dist_sq = norm_sq - 2 * torch.matmul(embeddings, embeddings.t()) + norm_sq.t()
    dist_sq = torch.clamp(dist_sq, min=0.0) 
    if squared:
        return dist_sq
    else:
        dist = torch.sqrt(dist_sq)
        mask = (dist_sq == 0.0).float()
        dist = dist + mask * 1e-16
        dist = dist * (1.0 - mask) 
        return dist

def batch_hard_triplet_loss(embeddings, labels, margin=0.2):
    labels = labels.unsqueeze(1)
    pdist = pairwise_distance(embeddings, squared=False) 
    mask_pos = (labels == labels.t())
    mask_neg = (labels != labels.t())

    max_pos = (pdist * mask_pos.float()).max(dim=1)[0]
    pdist_neg = pdist.clone()
    pdist_neg[mask_pos] = pdist.max() + 1.0 
    min_neg = pdist_neg.min(dim=1)[0]

    loss = torch.relu(max_pos - min_neg + margin)
    
    num_positive_triplets = (loss > 1e-7).float().sum()
    if num_positive_triplets == 0:
        return torch.tensor(0.0, device=embeddings.device)
        
    return loss.sum() / num_positive_triplets

# -------------------------
# build id2paths and label mapping
# -------------------------
def build_id2paths(root_dir):
    if not os.path.isdir(root_dir):
        return defaultdict(list)
    all_files = sorted(os.listdir(root_dir))
    image_files = [f for f in all_files if f.lower().endswith(('.jpg','.jpeg','.png'))]
    id2paths = defaultdict(list)
    for fname in image_files:
        if '_' not in fname: continue
        pid = fname.split('_')[0]
        if not pid.isdigit(): continue
        id2paths[int(pid)].append(os.path.join(root_dir, fname))
    return id2paths

def build_label_mapping(id2paths):
    unique_pids = sorted(id2paths.keys())
    pid2label = {pid: idx for idx, pid in enumerate(unique_pids)}
    return pid2label

# -------------------------
# main
# -------------------------
def main(
    dataset_root='/home/jovyan/dataset',
    backbone='resnet50', 
    use_cosface=False,
    P=32, K=4, 
    epochs=50,
    base_lr=1e-3, 
    warmup_epochs=5,
    triplet_weight=0.55, 
    margin=0.3, 
    embedding_size=512,
    dropout_p=0.3,
    resume=None
):
    print("--- Starting Metric Learning Training ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    train_root = os.path.join(dataset_root, 'train/images')
    val_root = os.path.join(dataset_root, 'val/images')

    # 1. Device Check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    # 2. Data Preparation
    id2paths_train = build_id2paths(train_root)
    id2paths_val = build_id2paths(val_root)
    pid2label_train = build_label_mapping(id2paths_train)
    pid2label_val = build_label_mapping(id2paths_val)
    num_classes = len(pid2label_train)
    print('Num classes:', num_classes)

    # 3. Model Initialization
    try:
        model = EmbeddingNet(embedding_size, pretrained=True).to(device)
        print('Model loaded to', device)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
        
    dropout = nn.Dropout(p=dropout_p)

    # Loss Initialization
    if use_cosface:
        metric_loss = CosFaceLoss(num_classes=num_classes, embedding_size=embedding_size,
                                  margin=0.35, scale=30.0, label_smoothing=0.0).to(device)
        print('Using CosFaceLoss')
    else:
        metric_loss = ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size,
                                  margin=0.30, scale=32.0, label_smoothing=0.05).to(device)
        print('Using ArcFaceLoss')

    # 4. Optimizer and Scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-4)
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # *** 修正: 使用 PyTorch 推薦的 'cuda' 參數 (如果 device==cuda) ***
    scaler = GradScaler(enabled=(device == 'cuda')) 

    ckpt_dir = os.path.join('models')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float('inf')
    es_counter = 0
    patience_es = 8

    batch_size = P * K
    steps_per_epoch = max(1, sum(len(v) for v in id2paths_train.values()) // batch_size)
    print(f'Training Batch Size: {batch_size}, Steps per Epoch: {steps_per_epoch}')

    if resume:
        try:
            ckpt = torch.load(resume, map_location=device)
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            else:
                model.load_state_dict(ckpt)
            print('Resumed weights from', resume)
        except Exception as e:
             print(f"[WARN] Failed to resume weights from {resume}: {e}")

    print("--- 5. Starting Training Loop ---")
    for epoch in range(epochs):
        model.train()
        metric_loss.train()
        epoch_loss = 0.0
        epoch_arc_loss = 0.0
        epoch_trip_loss = 0.0
        t0 = time.time()

        # warmup
        if epoch < warmup_epochs:
            lr = base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
            for g in optimizer.param_groups:
                g['lr'] = lr
        current_lr = optimizer.param_groups[0]['lr'] 

        for step in range(steps_per_epoch):
            # --- Training Sampler (PK) ---
            sel_ids = random.sample(list(id2paths_train.keys()), P)
            batch_pairs = []
            for pid in sel_ids:
                paths = id2paths_train[pid]
                if len(paths) >= K:
                    chosen = random.sample(paths, K)
                else:
                    chosen = random.choices(paths, k=K)
                for p in chosen:
                    batch_pairs.append((p, pid))
            
            imgs = []
            labels = []
            random.shuffle(batch_pairs)
            
            for p, pid in batch_pairs:
                img = pil_loader(p)
                img = train_transform(img)
                imgs.append(img.unsqueeze(0))
                labels.append(pid2label_train[pid])
            imgs = torch.cat(imgs, dim=0).to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            # *** 最終修正: 傳入 device_type 參數，解決 TypeError ***
            with autocast(device_type=device, enabled=(device == 'cuda')): 
                emb = model(imgs)
                emb = dropout(emb)
                loss_arc = metric_loss(emb, labels)
                
                loss_trip = batch_hard_triplet_loss(emb, labels, margin=margin)
                
                if loss_trip > 0: 
                    loss = (1.0 - triplet_weight) * loss_arc + triplet_weight * loss_trip
                else:
                    loss = loss_arc

            # --- Backward Pass ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * imgs.size(0)
            epoch_arc_loss += loss_arc.item() * imgs.size(0)
            epoch_trip_loss += loss_trip.item() * imgs.size(0)

        if epoch >= warmup_epochs:
            scheduler_cos.step()

        # validation
        model.eval()
        metric_loss.eval()
        val_running = 0.0
        val_steps = steps_per_epoch // 2
        val_data_size = val_steps * batch_size
        
        with torch.no_grad():
            val_ids = list(id2paths_val.keys())
            idx = 0
            for _ in range(val_steps):
                if idx + P > len(val_ids): idx = 0
                sel = val_ids[idx: idx+P]
                idx += P
                
                if len(sel) < P: sel = random.sample(val_ids, P) 

                batch_pairs = []
                for pid in sel:
                    paths = id2paths_val[pid]
                    if len(paths) >= K: chosen = random.sample(paths, K)
                    else: chosen = random.choices(paths, k=K)
                    for p in chosen: batch_pairs.append((p, pid))
                random.shuffle(batch_pairs) 
                imgs = []; labels = []
                for p, pid in batch_pairs:
                    img = pil_loader(p)
                    img = val_transform(img)
                    imgs.append(img.unsqueeze(0))
                    labels.append(pid2label_val.get(pid, 0))
                imgs = torch.cat(imgs, dim=0).to(device)
                labels = torch.tensor(labels, dtype=torch.long, device=device)
                
                with autocast(device_type=device, enabled=(device == 'cuda')): 
                    emb = model(imgs)
                    loss_arc = metric_loss(emb, labels)
                    loss_trip = batch_hard_triplet_loss(emb, labels, margin=margin)
                    
                    if loss_trip > 0:
                        loss = (1.0 - triplet_weight) * loss_arc + triplet_weight * loss_trip
                    else:
                        loss = loss_arc
                        
                val_running += loss.item() * imgs.size(0)
        
        val_loss = val_running / max(1, val_data_size)

        if epoch >= warmup_epochs:
             scheduler_plateau.step(val_loss)

        is_best = val_loss < best_val_loss
        
        train_loss_avg = epoch_loss / (steps_per_epoch * batch_size) 
        train_arc_loss_avg = epoch_arc_loss / (steps_per_epoch * batch_size)
        train_trip_loss_avg = epoch_trip_loss / (steps_per_epoch * batch_size)

        if is_best:
            best_val_loss = val_loss
            es_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'advanced_best_model2.pth'))
            print(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            es_counter += 1
            if es_counter >= patience_es:
                print('Early stopping triggered')
                break

        torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()},
                   os.path.join(ckpt_dir, f'checkpoint_epoch{epoch+1}.pth'))

        print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | Train Loss(avg): {train_loss_avg:.4f} | Arc Loss: {train_arc_loss_avg:.4f} | Trip Loss: {train_trip_loss_avg:.4f} | Val Loss: {val_loss:.4f} | Time: {(time.time()-t0):.1f}s")

    print('Finished Training.')

if __name__ == '__main__':
    main(dataset_root='/home/jovyan/dataset', backbone='resnet50', use_cosface=False,
         P=32, K=4, epochs=50, base_lr=1e-3, warmup_epochs=5, triplet_weight=0.55, margin=0.3)