from torch import nn
from torchvision import models

# 공통 이미지 특징 추출기
class ImageBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.model = models.efficientnet_b0(
            weights=(
                models.EfficientNet_B0_Weights.DEFAULT
                if pretrained else None
            )
        )

        self.feature_dim = self.model.classifier[1].in_features
        self.model.classifier = nn.Identity()

    def forward(self, image):
        return self.model(image)
