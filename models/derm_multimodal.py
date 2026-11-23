
import torch
import torch.nn as nn

from training.train_derm import DermClassifier


class DermMultimodalClassifier(nn.Module):
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

        for param in self.image_model.parameters():
            param.requires_grad = False

        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        # 학습 초기에는 이미지 모델과 같은 예측을 출력
        nn.init.zeros_(self.metadata_net[-1].weight)
        nn.init.zeros_(self.metadata_net[-1].bias)

    def forward(self, image, metadata):
        # 기존 이미지 모델은 고정
        self.image_model.eval()

        with torch.no_grad():
            image_logits = self.image_model(image)

        metadata_logits = self.metadata_net(metadata)

        return image_logits + metadata_logits
