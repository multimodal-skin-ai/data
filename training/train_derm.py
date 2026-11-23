import argparse
import json
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training.train import parse_soft, weighted_soft_ce, seed_everything, case_metrics, load_class_to_idx
from models.backbones.derm_foundation import DermClassifier


# 이미지 자체 대신 Derm Foundation이 미리 추출한 임베딩을 학습 샘플로 사용
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets, weights, case_ids):
        self.embeddings = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.targets = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
        self.weights = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32))
        self.case_ids = list(case_ids)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, i):
        return {"image": self.embeddings[i], "target": self.targets[i],
                "weight": self.weights[i], "case_id": self.case_ids[i]}


# 기존 33차원 Soft Label 불러오기
def get_soft_targets(df, old_mapping):
    result = []

    for raw in df["soft_label"]:
        v = parse_soft(raw, old_mapping)
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        result.append(np.asarray(v, dtype=np.float32))

    result = np.stack(result)

    if result.shape != (len(df), len(old_mapping)):
        raise ValueError(f"Unexpected soft label shape: {result.shape}")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Soft labels contain NaN, Inf or negative probabilities")
    if (result.sum(axis=1) <= 0).any():
        raise ValueError("Soft labels with zero total probability")

    return result / result.sum(axis=1, keepdims=True)


# ========================================
# 대상 클래스를 10개로 축소
# ========================================

# 필터링 / 기존 33-class Soft Label을 10개 질환 + other로 재구성
def filter_and_remap(df, old_targets, old_mapping, new_mapping):
    selected_classes = [
        name
        for name, idx in sorted(
            new_mapping.items(),
            key=lambda x: x[1]
        )
        if name != "other"
    ]

    missing = set(selected_classes) - set(old_mapping)
    if missing:
        raise ValueError(f"기존 class map에 없는 질환: {sorted(missing)}")

    selected_set = set(selected_classes)

    # 기존 33개 클래스에서 선정한 10개 질환의 인덱스
    indices = np.array(
        [old_mapping[name] for name in selected_classes],
        dtype=int
    )

    def check_conditions(row):
        # 원본 질환명 및 전문의 신뢰도 불러오기
        names = ast.literal_eval(
            row["dermatologist_skin_condition_on_label_name"]
        )
        confidences = ast.literal_eval(
            row["dermatologist_skin_condition_confidence"]
        )

        if len(names) != len(confidences):
            raise ValueError(
                f"질환명과 신뢰도 개수 불일치: {row['case_id']}"
            )

        # 선정한 10개 질환 중 하나 이상 포함
        cond1 = any(name in selected_set for name in names)

        # 최고 신뢰도를 가진 질환 중 하나 이상이 선정한 10개 질환에 포함
        max_conf = max(confidences)
        cond2 = any(
            name in selected_set and conf == max_conf
            for name, conf in zip(names, confidences)
        )

        return pd.Series([cond1, cond2])

    # 이미지별 행에 대해 필터링 조건 확인
    conditions = df.apply(check_conditions, axis=1)
    conditions.columns = ["cond1", "cond2"]

    cond1 = conditions["cond1"].to_numpy(dtype=bool)
    cond2 = conditions["cond2"].to_numpy(dtype=bool)

    mask = cond1 & cond2

    print(f"조건 1 충족: {df.loc[cond1, 'case_id'].nunique():,} cases")
    print(f"조건 2 충족: {df.loc[cond2, 'case_id'].nunique():,} cases")
    print(
        f"두 조건 모두: {df.loc[mask, 'case_id'].nunique():,} cases "
        f"/ {mask.sum():,} images"
    )

    if not mask.any():
        raise ValueError("필터링 후 남은 데이터가 없습니다.")

    # 같은 케이스의 이미지들이 모두 같은 필터링 결과를 갖는지 확인
    decisions = pd.DataFrame({
        "case_id": df["case_id"],
        "keep": mask
    })

    if (decisions.groupby("case_id")["keep"].nunique() > 1).any():
        raise ValueError("같은 케이스의 이미지별 필터링 결과가 다릅니다.")

    # 선택된 10개 질환의 Soft Label 확률 추출
    new_targets = old_targets[mask][:, indices].copy()

    # 선택되지 않은 질환의 확률을 other로 통합
    other_mass = np.maximum(
        0.0,
        1.0 - new_targets.sum(axis=1)
    )

    new_targets = np.column_stack(
        [new_targets, other_mass]
    ).astype(np.float32)

    new_targets /= new_targets.sum(axis=1, keepdims=True)

    return mask, new_targets



# 이미지별 예측을 case_id 단위로 평균 내어 Macro F1 / Accuracy 계산
def evaluate(model, loader, device, num_classes=11):
    model.eval()
    records = []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for cid, p, t in zip(
                batch["case_id"],
                probs,
                batch["target"].argmax(dim=1).cpu().tolist()
            ):
                records.append((cid, p, t))
              
    metrics = case_metrics(
        records,
        num_classes=num_classes,
        evaluation="validation"
    )

    return metrics


# ========================================
# 데이터 검증 -> 필터링 -> 학습 -> 검증 결과 저장
# ========================================
def main():
    parser = argparse.ArgumentParser()

    # 데이터 및 클래스 매핑 경로
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--original-class-map", type=Path, default=Path("config/class_to_idx.json"))
    parser.add_argument("--class-map", type=Path, default=Path("config/class_to_idx_top11.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("outputs/derm_top11"))
    args = parser.parse_args()
    seed_everything(args.seed)

    # 1. 기존 33개 및 신규 11개 클래스 매핑 불러오기
    old_mapping = load_class_to_idx(args.original_class_map)
    new_mapping = load_class_to_idx(args.class_map)

    if len(old_mapping) != 33:
        raise ValueError("기존 클래스 매핑은 33개여야 합니다.")
    if len(new_mapping) != 11 or new_mapping.get("other") != 10:
        raise ValueError("11개 클래스 매핑이 올바르지 않습니다.")

    df = pd.read_csv(
        args.data_csv,
        dtype={"case_id": str, "image_id": str}
    )

    required = {"case_id", "image_id", "split", "soft_label", "sample_weight", "dermatologist_skin_condition_on_label_name", "dermatologist_skin_condition_confidence"}

    if not required.issubset(df.columns):
        raise ValueError(f"필수 컬럼 누락: {required - set(df.columns)}")
    if df[list(required)].isna().any().any():
        raise ValueError("필수 컬럼에 결측치가 있습니다.")

    # 3. 기존 Derm Foundation 임베딩 불러오기
    with np.load(args.embeddings, allow_pickle=True) as data:
        embeddings = np.asarray(
            data["embeddings"],
            dtype=np.float32
        )

        if len(df) != len(embeddings):
            raise ValueError("CSV와 임베딩의 데이터 개수가 다릅니다.")

        # CSV와 임베딩의 순서 일치 여부 확인
        for column, key in [
            ("case_id", "case_ids"),
            ("image_id", "image_ids"),
            ("split", "splits")
        ]:
            if not np.array_equal(
                df[column].to_numpy(dtype=str),
                data[key].astype(str)
            ):
                raise ValueError(
                    f"CSV와 임베딩의 {column} 순서가 일치하지 않습니다."
                )

    if embeddings.ndim != 2 or embeddings.shape[1] != 6144:
        raise ValueError("Derm Foundation 임베딩 차원이 올바르지 않습니다.")
    if not np.isfinite(embeddings).all():
        raise ValueError("임베딩에 NaN 또는 Inf가 포함되어 있습니다.")

    # 4. 기존 Soft Label 불러오기
    old_targets = get_soft_targets(df, old_mapping)

    # 5. 원본 진단 기준 필터링 및 11개 클래스 Soft Label 생성
    keep, targets = filter_and_remap(
        df,
        old_targets,
        old_mapping,
        new_mapping
    )

    # 동일한 마스크로 CSV, 임베딩, Sample Weight 필터링
    weights = pd.to_numeric(
        df.loc[keep, "sample_weight"],
        errors="raise"
    ).to_numpy(dtype=np.float32)

    df = df.loc[keep].reset_index(drop=True)
    embeddings = embeddings[keep]

    if not np.isfinite(weights).all() or (
        (weights <= 0) | (weights > 1)
    ).any():
        raise ValueError("Sample Weight 값이 올바르지 않습니다.")

    # 6. 데이터 분할 및 케이스 중복 검증
    valid_splits = {"train", "validation", "test"}

    if not set(df["split"]).issubset(valid_splits):
        raise ValueError("지원하지 않는 split 값이 있습니다.")

    split_cases = {
        split: set(df.loc[df["split"] == split, "case_id"])
        for split in valid_splits
    }

    if (
        split_cases["train"] & split_cases["validation"]
        or split_cases["train"] & split_cases["test"]
        or split_cases["validation"] & split_cases["test"]
    ):
        raise ValueError("Train/Validation/Test 간 case_id 중복이 있습니다.")

    for split in ["train", "validation", "test"]:
        subset = df[df["split"] == split]
        print(
            f"{split}: "
            f"{subset['case_id'].nunique():,} cases / "
            f"{len(subset):,} images"
        )

    if not split_cases["train"] or not split_cases["validation"]:
        raise ValueError("Train 또는 Validation 데이터가 비어 있습니다.")

    # 7. Dataset 및 DataLoader 생성
    def loader_for(split, shuffle):
        mask = df["split"].eq(split).to_numpy()

        dataset = EmbeddingDataset(
            embeddings[mask],
            targets[mask],
            weights[mask],
            df.loc[mask, "case_id"].astype(str).tolist()
        )

        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle
        )

    train_loader = loader_for("train", True)
    val_loader = loader_for("validation", False)

    # 8. 모델 및 학습 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DermClassifier(
        input_dim=embeddings.shape[1],
        num_classes=len(new_mapping)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 9. 최고 모델 저장 및 Early Stopping 설정
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "derm_top11_best.pt"

    best, stale = float("-inf"), 0

    # 10. 모델 학습 및 검증
    for epoch in range(1, args.epochs + 1):
        model.train()

        total_numerator = 0.0
        total_denominator = 0.0

        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)

            logits = model(image)

            loss = weighted_soft_ce(logits, target, weight, class_weights=None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_numerator += loss.item() * weight.sum().item()
            total_denominator += weight.sum().item()

        train_loss = total_numerator / max(total_denominator, 1e-8)

        # Validation 케이스 단위 평가
        metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes=len(new_mapping)
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_macro_f1={metrics['macro_f1']:.4f}, "
            f"val_accuracy={metrics['accuracy']:.4f}"
        )

        # 클래스별 성능 출력
        idx_to_class = {v: k for k, v in new_mapping.items()}

        class_results = pd.DataFrame({
            "class_name": [
                idx_to_class[i]
                for i in range(len(new_mapping))
            ],
            "precision": metrics["class_precision"],
            "recall": metrics["class_recall"],
            "f1": metrics["class_f1"],
            "support": metrics["class_support"]
        })

        print("\n=== 클래스별 성능 ===")
        print(
            class_results
            .sort_values("f1")
            .to_string(index=False)
        )

        # Validation Macro F1 기준 최고 모델 저장
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            stale = 0

            torch.save({
                "model": model.state_dict(),
                "class_map": new_mapping,
                "epoch": epoch,
                "val_macro_f1": best,
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                }
            }, checkpoint)

        else:
            stale += 1

            if stale >= args.patience:
                print("Early stopping")
                break

    print(
        f"Training completed. "
        f"Best Macro F1: {best:.4f}, "
        f"Checkpoint: {checkpoint}"
    )


if __name__ == "__main__":
    main()
