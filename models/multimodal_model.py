import torch
from torch import nn
from .backbone import ImageBackbone


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        symptom_dim,
        pretrained=False,
        freeze_backbone=True
    ):
        super().__init__()

        self.backbone = ImageBackbone(pretrained)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.symptom_net = nn.Sequential(
            nn.Linear(symptom_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.feature_dim + 32, num_classes)
        )

    def forward(self, image, symptoms):
        image_features = self.backbone(image)
        symptom_features = self.symptom_net(symptoms)

        features = torch.cat(
            [image_features, symptom_features],
            dim=1
        )

        return self.head(features)

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_backbone:
            self.backbone.eval()

        return self
