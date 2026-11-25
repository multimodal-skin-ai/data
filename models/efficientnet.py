from torch import nn
from torchvision import models

# 이미지 단독 모델과 멀티모달 모델에서 공통으로 사용하는 특징 추출기
class ImageBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.model = models.efficientnet_b3(
            weights=(
                models.EfficientNet_B3_Weights.DEFAULT
                if pretrained else None
            )
        )

        self.feature_dim = self.model.classifier[1].in_features # 1536
        self.model.classifier = nn.Identity() # 기존 분류기 제거

    def forward(self, image):
        return self.model(image)