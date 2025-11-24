# src/extractor.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from model import EmbeddingNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

class SimpleFolderDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = [f for f in sorted(os.listdir(root)) if f.lower().endswith(('.jpg','.jpeg','.png'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        path = os.path.join(self.root, fn)
        img = Image.open(path).convert('RGB')
        return transform(img), fn


def extract_embeddings(model_path, image_root, out_path, batch_size=64, device='cuda'):
    device = device if torch.cuda.is_available() else 'cpu'
    model = EmbeddingNet(512).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.eval()

    ds = SimpleFolderDataset(image_root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    all_emb = []
    all_files = []
    with torch.no_grad():
        for imgs, fns in loader:
            imgs = imgs.to(device)
            emb = model(imgs)
            all_emb.append(emb.cpu().numpy())
            all_files.extend(fns)
    all_emb = np.vstack(all_emb)
    np.savez_compressed(out_path, embeddings=all_emb, filenames=np.array(all_files))
    print(f"Saved embeddings to {out_path} with shape {all_emb.shape}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to trained model .pth')
    parser.add_argument('--images', required=True, help='root folder with images')
    parser.add_argument('--out', required=True, help='output .npz path')
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    extract_embeddings(args.model, args.images, args.out, batch_size=args.batch)
