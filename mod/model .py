# src/model.py
import torch.nn as nn
import torchvision.models as models


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=512, pretrained=True):
        super().__init__()
        # torchvision weight API
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        num_features = backbone.fc.in_features

        # remove original fc and avgpool is included; we keep everything except fc
        features = list(backbone.children())[:-1]  # up to avgpool
        self.backbone = nn.Sequential(*features)  # outputs [B, num_features, 1, 1]

        self.embedding = nn.Linear(num_features, embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.bn = nn.BatchNorm1d(embedding_size)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.bn(x)
        return x
