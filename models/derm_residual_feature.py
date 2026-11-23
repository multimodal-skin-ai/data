
import torch
import torch.nn as nn
from training.train_derm import DermClassifier


class DermResidualFeature(nn.Module):
    def __init__(self, metadata_dim, checkpoint_path,
                 num_classes=11, alpha=0.5):
        super().__init__()

        self.alpha = alpha

        self.image_model = DermClassifier(
            input_dim=6144,
            num_classes=num_classes
        )

        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
        self.image_model.load_state_dict(ckpt["model"])

        # 기존 이미지 모델 전체 고정
        for p in self.image_model.parameters():
            p.requires_grad = False

        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 이미지 특징 + 메타데이터 특징
        self.correction_net = nn.Sequential(
            nn.Linear(256 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

        # 초기 보정값을 0으로 설정
        nn.init.zeros_(self.correction_net[-1].weight)
        nn.init.zeros_(self.correction_net[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.image_model.eval()
        return self

    def forward(self, image, metadata):
        self.image_model.eval()

        with torch.no_grad():
            image_feat = self.image_model.classifier[:4](image)
            image_logits = self.image_model.classifier[4](image_feat)

        metadata_feat = self.metadata_net(metadata)

        fused = torch.cat(
            [image_feat, metadata_feat],
            dim=1
        )

        correction = self.correction_net(fused)

        return image_logits + self.alpha * correction
