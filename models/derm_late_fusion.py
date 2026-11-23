
import torch
import torch.nn as nn

from training.train_derm import DermClassifier


class DermLateFusion(nn.Module):
    def __init__(self, metadata_dim, checkpoint_path, num_classes=11):
        super().__init__()

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

        # 이미지 모델 고정
        for p in self.image_model.parameters():
            p.requires_grad = False

        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 256차원 이미지 + 64차원 메타데이터
        self.fusion = nn.Linear(256 + 64, num_classes)

        # 기존 이미지 분류기의 마지막 층 복사
        with torch.no_grad():
            old_head = self.image_model.classifier[4]

            self.fusion.weight.zero_()
            self.fusion.weight[:, :256].copy_(old_head.weight)
            self.fusion.bias.copy_(old_head.bias)

    def train(self, mode=True):
        super().train(mode)
        # 고정된 이미지 분기의 Dropout 비활성화
        self.image_model.eval()
        return self

    def forward(self, image, metadata):
        self.image_model.eval()

        with torch.no_grad():
            image_feat = self.image_model.classifier[:4](image)

        metadata_feat = self.metadata_net(metadata)

        fused = torch.cat(
            [image_feat, metadata_feat],
            dim=1
        )

        return self.fusion(fused)
