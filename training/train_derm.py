import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from training.train import (
    parse_soft,
    weighted_soft_ce,
    compute_class_weights,
    case_metrics,
    seed_everything,
)

class DermEmbeddingDataset(Dataset):
    def __init__(self, frame, embeddings, mapping):
        self.df = frame.reset_index(drop=True)
        self.embeddings = embeddings
        self.mapping = mapping

        assert len(self.df) == len(self.embeddings)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        return {
            "image": torch.tensor(
                self.embeddings[index],
                dtype=torch.float32
            ),
            "target": parse_soft(
                row["soft_label"],
                self.mapping
            ),
            "weight": torch.tensor(
                float(row["sample_weight"]),
                dtype=torch.float32
            ),
            "case_id": str(row["case_id"])
        }

class DermClassifier(nn.Module):
    def __init__(self, num_classes=33):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(6144),
            nn.Linear(6144, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, symptoms=None):
        return self.classifier(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, default=Path("config/class_to_idx.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weighting", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/derm"))
    args = parser.parse_args()

    seed_everything(args.seed)

    df = pd.read_csv(args.data_csv, dtype={"case_id": str, "image_id": str})

    with open(args.class_map, encoding="utf-8") as f:
        mapping = json.load(f)

    data = np.load(args.embeddings)
    embeddings = data["embeddings"]

    # CSV와 임베딩의 순서가 일치하는지 확인
    assert len(df) == len(embeddings)
    assert np.array_equal(
        df["case_id"].astype(str).to_numpy(),
        data["case_ids"].astype(str)
    )
    assert np.array_equal(
        df["image_id"].astype(str).to_numpy(),
        data["image_ids"].astype(str)
    )

    train_mask = df["split"].eq("train").to_numpy()
    val_mask = df["split"].eq("validation").to_numpy()

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()

    train_dataset = DermEmbeddingDataset(
        train_df,
        embeddings[train_mask],
        mapping
    )

    val_dataset = DermEmbeddingDataset(
        val_df,
        embeddings[val_mask],
        mapping
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DermClassifier(num_classes=len(mapping)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr
    )

    class_weights = (
        compute_class_weights(train_df, mapping).to(device)
        if args.class_weighting else None
    )

    best_f1 = -1.0
    stale = 0

    args.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_num = 0.0
        train_den = 0.0

        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)

            logits = model(image)

            loss = weighted_soft_ce(
                logits,
                target,
                weight,
                class_weights=class_weights
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_num += loss.item() * weight.sum().item()
            train_den += weight.sum().item()

        train_loss = train_num / max(train_den, 1e-8)

        model.eval()
        records = []

        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                target = batch["target"].to(device)

                logits = model(image)
                probs = torch.softmax(logits, dim=1)

                for cid, prob, label in zip(
                    batch["case_id"],
                    probs.cpu().numpy(),
                    target.argmax(dim=1).cpu().numpy()
                ):
                    records.append((cid, prob, int(label)))

        metrics = case_metrics(
            records,
            num_classes=len(mapping),
            evaluation="validation"
        )

        f1 = metrics["macro_f1"]

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_macro_f1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            stale = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1": f1,
                    "config": vars(args),
                },
                args.out / "derm_best.pt"
            )
        else:
            stale += 1

        if stale >= args.patience:
            print("Early stopping")
            break

    print(f"Best Validation Macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()