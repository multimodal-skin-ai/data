from torch import nn
from .backbone import ImageBackbone

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.backbone = ImageBackbone(pretrained)

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.feature_dim, num_classes)
        )

    def forward(self, image, symptoms=None):
        features = self.backbone(image)
        return self.head(features)
