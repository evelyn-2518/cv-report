import os
import sys

# =========================================================
# ⚠️【關鍵修復】防止 macOS / Linux 多執行緒衝突導致死鎖
# 必須放在 import numpy 或 torch 之前執行
# =========================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  # 降維用
import matplotlib.pyplot as plt
import time
from PIL import Image
from collections import OrderedDict
from typing import List, Tuple, Dict

# =========================================================
# 區域 0: 環境與路徑設定
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if getattr(sys, 'frozen', False):
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 設定數據集根目錄
DATASET_ROOT = "/Users/winny/Desktop/電腦視覺/cv_final_project/dataset" 

# 2. Advanced 模型路徑
MODEL_FILENAME = '/Users/winny/Desktop/電腦視覺/cv_final_project/advanced_best_model2.pth'
MODEL_PATH = MODEL_FILENAME 

# 3. 輸出圖片路徑 (改名以區別)
TSNE_IMAGE_PATH = os.path.join(SCRIPT_DIR, 't_sne_advanced_pca_fix.png')

# 4. 可視化參數 (與 Baseline 保持一致以方便比較)
MAX_TSNE_SAMPLES = 3000   # 抽樣點數
PCA_COMPONENTS = 50       # PCA 先降到 50 維
TSNE_PERPLEXITY = 30      # t-SNE 參數

# =========================================================
# 區域 1: Dataset 與 Transforms
# =========================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

VAL_TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE), 
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def build_label_map(root_dir: str) -> Tuple[List[str], dict]:
    all_item_ids = set()
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(root_dir, split, "images")
        if not os.path.exists(split_dir):
            continue
        try:
            files = os.listdir(split_dir)
            for img_file in files:
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if "_" in img_file:
                        item_id = img_file.split("_")[0]
                        all_item_ids.add(item_id)
        except OSError as e:
            print(f"無法讀取目錄 {split_dir}: {e}")

    sorted_ids = sorted(list(all_item_ids))
    item_to_idx = {item_id: idx for idx, item_id in enumerate(sorted_ids)}
    return sorted_ids, item_to_idx

class FolderBasedRetrievalDataset(Dataset):
    def __init__(self, images_dir: str, item_to_idx: dict, transform=None):
        self.images_dir = images_dir
        self.item_to_idx = item_to_idx
        self.transform = transform

        if not os.path.exists(images_dir):
            raise ValueError(f"找不到圖片目錄: {images_dir}")

        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        self.valid_files = []
        for f in self.image_files:
            if "_" in f:
                item_id = f.split("_")[0]
                if item_id in self.item_to_idx:
                    self.valid_files.append(f)
        
        print(f"資料夾 {os.path.basename(images_dir)}: 載入 {len(self.valid_files)} 張有效圖片")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_name = self.valid_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        item_id = img_name.split("_")[0]
        label = self.item_to_idx[item_id] 

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), label
        
        if self.transform:
            image = self.transform(image)
        return image, label

def setup_data_loaders():
    print(f"🚀 正在載入數據集: {DATASET_ROOT}")
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 錯誤: 找不到路徑 {DATASET_ROOT}")
        sys.exit(1)

    print("正在建立標籤映射...")
    ITEM_ID_LIST, ITEM_TO_IDX = build_label_map(DATASET_ROOT)
    
    TRAIN_IMAGES_DIR = os.path.join(DATASET_ROOT, "train", "images")
    VAL_IMAGES_DIR = os.path.join(DATASET_ROOT, "val", "images")

    train_dataset = FolderBasedRetrievalDataset(TRAIN_IMAGES_DIR, ITEM_TO_IDX, VAL_TEST_TRANSFORMS)
    val_dataset = FolderBasedRetrievalDataset(VAL_IMAGES_DIR, ITEM_TO_IDX, VAL_TEST_TRANSFORMS)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, val_loader

# =========================================================
# 區域 2: Advanced 模型載入邏輯 (保留您原本的寫法)
# =========================================================

def load_advanced_model(path: str) -> nn.Module:
    print(f"正在載入 Advanced 模型: {path}")
    
    # 1. 建立標準 ResNet50
    model = models.resnet50(weights=None)
    model.fc = nn.Identity()

    if not os.path.exists(path):
        print(f"❌ 錯誤：找不到檔案 {path}")
        return model.to(DEVICE)

    try:
        # 2. 載入權重
        checkpoint = torch.load(path, map_location=DEVICE)
        state_dict = checkpoint['state_dict'] if (isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint

        # 3. 權重映射：將 backbone.xxx 轉為標準 ResNet 格式
        new_state_dict = OrderedDict()
        
        # 映射規則
        mapping = {
            'backbone.0': 'conv1',
            'backbone.1': 'bn1',
            'backbone.4': 'layer1',
            'backbone.5': 'layer2',
            'backbone.6': 'layer3',
            'backbone.7': 'layer4'
        }

        for k, v in state_dict.items():
            new_key = k
            for old_prefix, new_prefix in mapping.items():
                if k.startswith(old_prefix):
                    new_key = k.replace(old_prefix, new_prefix)
                    break
            
            # 忽略分類層
            if 'fc' in new_key or 'classifier' in new_key or 'head' in new_key:
                continue

            new_state_dict[new_key] = v

        # 4. 載入修正後的權重
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Advanced 權重映射與載入完成。")
        
    except Exception as e:
        print(f"❌ 載入失敗: {e}")

    model = model.to(DEVICE)
    model.eval()
    return model

def extract_features(loader, model):
    features_list, labels_list = [], []
    model.eval()
    print(f"提取特徵中 (Dataset size: {len(loader.dataset)})...")
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            # L2 正規化
            feats = F.normalize(feats, p=2, dim=1)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.concatenate(features_list), np.concatenate(labels_list)

def main():
    # 1. 數據與模型
    try:
        gallery_loader, query_loader = setup_data_loaders()
        model = load_advanced_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return

    # 2. 提取特徵
    print("\n--- 提取 Query 特徵 ---")
    query_feats, query_labels = extract_features(query_loader, model)
    print("\n--- 提取 Gallery 特徵 ---")
    gallery_feats, gallery_labels = extract_features(gallery_loader, model)

    # 3. 計算 Recall (僅作確認用)
    print("\n計算相似度矩陣...")
    sim_matrix = np.dot(query_feats, gallery_feats.T)
    
    if query_feats.shape == gallery_feats.shape:
        print("⚠️ 執行 Self-Masking (排除自身)...")
        np.fill_diagonal(sim_matrix, -np.inf)
    
    ks = [1, 5, 10, 20]
    recall_counts = {k: 0 for k in ks}
    gallery_dict = {}
    for i, label in enumerate(gallery_labels):
        if label not in gallery_dict: gallery_dict[label] = []
        gallery_dict[label].append(i)

    print("正在評估 Recall...")
    for i in tqdm(range(len(query_labels))):
        q_label = query_labels[i]
        target_indices = gallery_dict.get(q_label, [])
        if not target_indices: continue

        top_indices = np.argsort(sim_matrix[i])[-20:][::-1]
        for k in ks:
            if np.isin(top_indices[:k], target_indices).any():
                recall_counts[k] += 1

    print("\n========== Advanced 模型評估結果 ==========")
    for k in ks:
        print(f"Recall@{k}: {recall_counts[k] / len(query_labels) * 100:.2f}%")
    print("===========================================")

    # 4. t-SNE 可視化 (使用 PCA 加速版)
    print("\n準備進行可視化...")
    all_feats = np.concatenate([query_feats, gallery_feats])
    all_labels = np.concatenate([query_labels, gallery_labels])
    
    # 抽樣
    if len(all_feats) > MAX_TSNE_SAMPLES:
        print(f"隨機抽樣 {MAX_TSNE_SAMPLES} 筆資料...")
        np.random.seed(42)
        idx = np.random.choice(len(all_feats), MAX_TSNE_SAMPLES, replace=False)
        all_feats = all_feats[idx]
        all_labels = all_labels[idx]

    # NaN 檢查
    if np.isnan(all_feats).any():
        print("⚠️ 發現 NaN 數值，正在修復...")
        all_feats = np.nan_to_num(all_feats)

    # PCA 降維 (2048 -> 50)
    print(f"正在執行 PCA (2048 -> {PCA_COMPONENTS})...")
    time_pca = time.time()
    try:
        pca = PCA(n_components=PCA_COMPONENTS, svd_solver='randomized', random_state=42)
        all_feats_pca = pca.fit_transform(all_feats)
        print(f"PCA 完成 (耗時 {time.time() - time_pca:.2f}s)")
    except Exception as e:
        print(f"❌ PCA 失敗: {e}")
        return

    # t-SNE
    print(f"正在執行 t-SNE (Advanced Model, Perp={TSNE_PERPLEXITY})...")
    time_tsne = time.time()
    
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, max_iter=1000, 
                init='pca', random_state=42, n_jobs=-1, verbose=1)
    embeddings_2d = tsne.fit_transform(all_feats_pca)
    print(f"t-SNE 完成 (耗時 {time.time() - time_tsne:.2f}s)")

    # 繪圖
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=all_labels, cmap='Spectral', s=8, alpha=0.6)
    plt.title(f"t-SNE (Advanced, PCA={PCA_COMPONENTS}, Perp={TSNE_PERPLEXITY})")
    plt.axis('off')
    plt.savefig(TSNE_IMAGE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 圖表已儲存至: {TSNE_IMAGE_PATH}")

if __name__ == "__main__":
    main()
