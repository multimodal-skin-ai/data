import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.backbones.derm_foundation import DermClassifier
from training.train import case_metrics, load_class_to_idx
from training.train_derm import (
    EmbeddingDataset,
    get_soft_targets,
    filter_and_remap,
)
from evaluation.evaluate import build_class_report


# ============================================================
# 1. CSV와 Derm Foundation 임베딩 불러오기 및 검증
# ============================================================
def load_data(data_csv, embeddings_path):

    df = pd.read_csv(data_csv, dtype={"case_id": str, "image_id": str,})

    required = {
        "case_id",
        "image_id",
        "split",
        "soft_label",
        "sample_weight",
        "dermatologist_skin_condition_on_label_name",
        "dermatologist_skin_condition_confidence",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("필수 컬럼에 결측치가 있습니다.")

    with np.load(embeddings_path, allow_pickle=True) as data:
        embeddings = np.asarray(
            data["embeddings"],
            dtype=np.float32
        )

        if len(df) != len(embeddings):
            raise ValueError("CSV와 임베딩의 데이터 개수가 다릅니다.")

        # CSV와 임베딩의 행 순서 검증
        for column, key in [
            ("case_id", "case_ids"),
            ("image_id", "image_ids"),
            ("split", "splits"),
        ]:
            if not np.array_equal(
                df[column].to_numpy(dtype=str),
                data[key].astype(str)
            ):
                raise ValueError(
                    f"CSV와 임베딩의 {column} 순서가 "
                    "일치하지 않습니다."
                )

    if embeddings.ndim != 2 or embeddings.shape[1] != 6144:
        raise ValueError(
            f"임베딩 차원이 올바르지 않습니다: "
            f"{embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "임베딩에 NaN 또는 Inf가 포함되어 있습니다."
        )

    print(
        f"전체 데이터: {len(df):,} images / "
        f"{df['case_id'].nunique():,} cases"
    )

    return df, embeddings


# ============================================================
# 2. 학습과 동일한 필터링 및 Test 데이터 구성
# ============================================================
def prepare_test_data(
    df,
    embeddings,
    old_mapping,
    new_mapping,
    batch_size,
):

    # 기존 33차원 Soft Label 불러오기
    old_targets = get_soft_targets(df, old_mapping)

    # 학습과 동일한 필터링 및 11차원 재구성
    keep, targets = filter_and_remap(
        df,
        old_targets,
        old_mapping,
        new_mapping
    )

    # CSV, 임베딩, 타깃에 동일한 마스크 적용
    df = df.loc[keep].reset_index(drop=True)
    embeddings = embeddings[keep]

    weights = pd.to_numeric(
        df["sample_weight"],
        errors="raise"
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(weights).all() or (
        (weights <= 0) | (weights > 1)
    ).any():
        raise ValueError("Sample Weight 값이 올바르지 않습니다.")

    # Test split 선택
    test_mask = df["split"].eq("test").to_numpy()

    test_df = df.loc[test_mask].reset_index(drop=True)
    test_embeddings = embeddings[test_mask]
    test_targets = targets[test_mask]
    test_weights = weights[test_mask]

    if test_df.empty:
        raise ValueError("필터링 후 Test 데이터가 없습니다.")

    split_cases = {
        split: set(
            df.loc[df["split"] == split, "case_id"]
        )
        for split in ["train", "validation", "test"]
    }

    if (
        split_cases["train"] & split_cases["validation"]
        or split_cases["train"] & split_cases["test"]
        or split_cases["validation"] & split_cases["test"]
    ):
        raise ValueError("Train/Validation/Test 간 case_id 중복이 있습니다.")

    dataset = EmbeddingDataset(
        embeddings=test_embeddings,
        targets=test_targets,
        weights=test_weights,
        case_ids=test_df["case_id"].astype(str).tolist()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print("\n===== TEST DATA =====")
    print(f"Cases: {test_df['case_id'].nunique():,}")
    print(f"Images: {len(test_df):,}")

    return loader, test_df


# ============================================================
# 3. Test
# ============================================================
def evaluate_test(model, loader, device, num_classes):
    model.eval()

    records = []
    with torch.no_grad():

        for batch in loader:
            embeddings = batch["image"].to(device)
            logits = model(embeddings)
            probs = torch.softmax(
                logits,
                dim=1
            ).cpu().numpy()

            targets = batch["target"].argmax(
                dim=1
            ).cpu().tolist()

            for case_id, prob, target in zip(
                batch["case_id"],
                probs,
                targets
            ):
                records.append(
                    (case_id, prob, target)
                )

    # 기존 프로젝트의 case 단위 평가 함수 재사용
    metrics = case_metrics(
        records,
        num_classes=num_classes,
        evaluation="test"
    )

    return metrics


# ============================================================
# 4. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-class-map", type=Path, default=Path("config/class_to_idx.json"))
    parser.add_argument("--class-map", type=Path, default=Path("config/class_to_idx_top11.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 클래스 매핑 불러오기
    old_mapping = load_class_to_idx(args.original_class_map)
    new_mapping = load_class_to_idx(args.class_map)

    if len(old_mapping) != 33:
        raise ValueError("기존 클래스 매핑은 33개여야 합니다.")

    if len(new_mapping) != 11 or new_mapping.get("other") != 10:
        raise ValueError("11개 클래스 매핑이 올바르지 않습니다.")


    # 2. 체크포인트 불러오기
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["class_map"] != new_mapping:
        raise ValueError("체크포인트와 평가용 클래스 매핑이 일치하지 않습니다.")

    
    # 3. CSV 및 임베딩 불러오기
    df, embeddings = load_data(args.data_csv, args.embeddings)


    # 4. Test DataLoader 생성
    test_loader, test_df = prepare_test_data(
        df=df,
        embeddings=embeddings,
        old_mapping=old_mapping,
        new_mapping=new_mapping,
        batch_size=args.batch_size
    )

    # 5. 모델 생성 및 가중치 로드
    model = DermClassifier(
        input_dim=embeddings.shape[1],
        num_classes=len(new_mapping)
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model = model.to(device)
    model.eval()

    print("\n===== MODEL =====")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Best epoch: {checkpoint.get('epoch', 'N/A')}")
    print(
        f"Validation Macro F1: "
        f"{checkpoint.get('val_macro_f1', float('nan')):.4f}"
    )

    # 6. Test 평가
    metrics = evaluate_test(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=len(new_mapping)
    )

    print("\n===== TEST RESULTS =====")
    print(f"Cases: {metrics['cases']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(
        f"Macro F1 classes: "
        f"{metrics['macro_num_classes']}"
    )


    # 7. 클래스별 성능
    class_report = build_class_report(
        metrics,
        new_mapping
    )

    print("\n===== CLASS-WISE RESULTS =====")
    print(
        class_report.to_string(index=False)
    )


    # 8. 결과 CSV 저장
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = args.checkpoint.stem

    summary_df = pd.DataFrame([{
        "model": "derm_foundation",
        "checkpoint": checkpoint_name,
        "cases": metrics["cases"],
        "images": len(test_df),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_num_classes": metrics["macro_num_classes"],
    }])

    summary_path = (
        args.output_dir
        / f"{checkpoint_name}_test_results.csv"
    )

    classwise_path = (
        args.output_dir
        / f"{checkpoint_name}_classwise_results.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    class_report.to_csv(
        classwise_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== SAVED =====")
    print(f"Test results: {summary_path}")
    print(f"Class-wise results: {classwise_path}")


if __name__ == "__main__":
    main()
