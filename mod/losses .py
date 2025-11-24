# src/losses.py (Final Fix for AMP RuntimeError)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    # ... (__init__ 保持不變，參數已修正) ...
    def __init__(self, num_classes, embedding_size, margin=0.30, scale=32.0, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes 
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        self.label_smoothing = float(label_smoothing)

        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin 
        
    def forward(self, embeddings, labels):
        """
        embeddings: (B, emb_size)
        labels: (B,) long
        """
        # normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)

        # logits: cosine similarity (B, C)
        logits = torch.matmul(embeddings, W.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        
        # 獲取目標數據類型 (在 autocast 內通常是 torch.float16)
        target_dtype = logits.dtype 

        # gather target logits
        batch_idx = torch.arange(0, embeddings.size(0), dtype=torch.long, device=labels.device)
        target_logits = logits[batch_idx, labels]  # (B,)

        # 確保 sin_theta 至少為 1e-6 (避免數值問題)
        sin_theta = torch.sqrt(1.0 - target_logits.pow(2)).clamp(min=1e-6) 
        cos_theta_m = target_logits * self.cos_m - sin_theta * self.sin_m

        cond = target_logits - self.th
        final_target_logits = torch.where(cond > 0, cos_theta_m, target_logits - self.mm)

        # construct new logits
        logits_modified = logits.clone()
        
        # *** 最終修正: 將 final_target_logits 轉換為目標數據類型 (Half/float16) ***
        logits_modified[batch_idx, labels] = final_target_logits.to(target_dtype)

        # scale
        logits_modified = logits_modified * self.scale

        # ... (Label smoothing 邏輯保持不變) ...
        # 如果要計算 Cross Entropy (通常是 F.cross_entropy)，則需要將 logits 轉回 float32
        
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = self.label_smoothing
                off_value = smooth / (self.num_classes - 1)
                on_value = 1.0 - smooth
                y = torch.full_like(logits_modified, off_value)
                y[batch_idx, labels] = on_value
            # For log_softmax/CrossEntropy, it's safer to cast back to float32
            loss = -torch.sum(y.float() * F.log_softmax(logits_modified.float(), dim=1), dim=1).mean()
        else:
            # For F.cross_entropy, input should be float32
            loss = F.cross_entropy(logits_modified.float(), labels)

        return loss