# src/losses_cosface.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosFaceLoss(nn.Module):
    """
    CosFace loss implementation.
    Can be used interchangeably with ArcFaceLoss.
    """
    def __init__(self, num_classes, embedding_size, margin=0.35, scale=30.0, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        self.label_smoothing = float(label_smoothing)

        # weight for fully connected layer
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)

        logits = torch.matmul(embeddings, W.t()).clamp(-1 + 1e-7, 1 - 1e-7)

        batch_idx = torch.arange(0, embeddings.size(0), device=labels.device)
        logits_modified = logits.clone()
        logits_modified[batch_idx, labels] -= self.margin
        logits_modified *= self.scale

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = self.label_smoothing
                off_value = smooth / (self.num_classes - 1)
                on_value = 1.0 - smooth
                y = torch.full_like(logits_modified, off_value)
                y[batch_idx, labels] = on_value
            loss = -torch.sum(y * F.log_softmax(logits_modified, dim=1), dim=1).mean()
        else:
            loss = F.cross_entropy(logits_modified, labels)
        return loss
