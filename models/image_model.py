from torch import nn
from .backbones.efficientnet import EfficientNetBackbone

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.backbone = EfficientNetBackbone(pretrained)

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.feature_dim, num_classes)
        )

    def forward(self, image, symptoms=None):
        features = self.backbone(image)
        return self.head(features)