from torch import nn

class DermClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)