# src/dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# *** 調整 RandomResizedCrop 尺度和 ColorJitter 參數 ***
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    # 讓裁剪尺度更廣，增加樣本多樣性，有助於泛化
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)), 
    transforms.RandomHorizontalFlip(),
    # 增加顏色擾動強度，讓模型對光照變化更健壯
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    transforms.RandomRotation(15), # 旋轉角度也稍增
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

VAL_TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


class FolderDataset(Dataset):
    """
    FolderDataset(root, train=True/False, id2idx=None)

    - root: path to folder containing images (e.g. dataset/train/images)
    - train: if True build id2idx from this folder; if False expect id2idx passed (or will build but warn)
    - id2idx: mapping from item_id (int) -> class idx; for val/test pass train.id2idx
    """

    def __init__(self, root, train=True, id2idx=None):
        self.root = root
        self.train = train
        self.transform = TRAIN_TRANSFORMS if train else VAL_TEST_TRANSFORMS

        if not os.path.isdir(root):
            raise ValueError(f"Dataset root not found: {root}")

        all_files = sorted(os.listdir(root))
        image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        raw_ids = []
        paths = []
        for name in image_files:
            # defensive checks
            if "_" not in name:
                # print(f"[WARN] skipping (no underscore): {name}") # 減少 log 輸出
                continue
            prefix = name.split("_")[0]
            if not prefix.isdigit():
                # print(f"[WARN] skipping (non-numeric prefix): {name}") # 減少 log 輸出
                continue

            full_path = os.path.join(root, name)
            # verify image can be opened
            try:
                with Image.open(full_path) as img:
                    img.verify()
            except Exception as e:
                # print(f"[WARN] corrupted/unreadable image skipped: {name} ({e})") # 減少 log 輸出
                continue

            raw_ids.append(int(prefix))
            paths.append(full_path)

        if len(paths) == 0:
            raise ValueError(f"No valid images found in {root}")

        self.paths = paths
        self.raw_ids = raw_ids

        # build or use id2idx
        if train:
            unique_ids = sorted(set(raw_ids))
            self.id2idx = {id_: i for i, id_ in enumerate(unique_ids)}
        else:
            if id2idx is None:
                # fallback: build from val set (not recommended)
                unique_ids = sorted(set(raw_ids))
                self.id2idx = {id_: i for i, id_ in enumerate(unique_ids)}
                print("[WARN] val dataset built its own id2idx (should pass train.id2idx)")
            else:
                self.id2idx = id2idx

        # map labels; skip any id not in id2idx (only possible for val if mapping not aligned)
        labels = []
        clean_paths = []
        for p, rid in zip(self.paths, self.raw_ids):
            if rid not in self.id2idx:
                # skip val items not seen in train
                if not train:
                    continue
                else:
                    raise RuntimeError(f"train id missing in id2idx: {rid}")
            labels.append(self.id2idx[rid])
            clean_paths.append(p)

        self.paths = clean_paths
        self.labels = labels

        # expose mapping and counts
        self.id2idx = self.id2idx
        self.idx2id = {v: k for k, v in self.id2idx.items()}

        print(f"Loaded {len(self.paths)} images from {root}")
        print(f"Detected {len(self.id2idx)} item_ids.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label