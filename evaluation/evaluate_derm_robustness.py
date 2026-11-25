import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.backbones.derm_foundation import DermClassifier
from training.train import case_metrics, load_class_to_idx
from training.train_derm import EmbeddingDataset, get_soft_targets, filter_and_remap
from evaluation.evaluate import build_class_report

CONDITIONS = ("original", "lowres", "blur", "dark")
KEYS = ["case_id", "image_id", "split"]


# ============================================================
# 1. 임베딩 불러오기 및 CSV와의 데이터 정합성 검증
# ============================================================
def load_aligned_embeddings(path, expected_df):
    with np.load(path, allow_pickle=True) as npz:
        required = {"embeddings", "case_ids", "image_ids", "splits"}
        if not required.issubset(npz.files):
            raise ValueError(f"{path}: missing NPZ keys: {required - set(npz.files)}")
        embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
        meta = pd.DataFrame({
            "case_id": npz["case_ids"].astype(str),
            "image_id": npz["image_ids"].astype(str),
            "split": npz["splits"].astype(str),
        })

    # 임베딩 차원 및 유효성 검증
    if embeddings.ndim != 2 or embeddings.shape[1] != 6144:
        raise ValueError(f"{path}: expected embedding shape (N, 6144), got {embeddings.shape}")
    if len(meta) != len(embeddings) or not np.isfinite(embeddings).all():
        raise ValueError(f"{path}: invalid embedding length or non-finite values")
    if meta.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate case_id/image_id/split keys")

    # CSV와 임베딩의 이미지 개수 및 중복 여부 검증
    expected = expected_df[KEYS].astype(str).reset_index(drop=True)
    if expected.duplicated(KEYS).any():
        raise ValueError("CSV contains duplicate case_id/image_id/split keys")
    if len(meta) != len(expected):
        raise ValueError(f"{path}: expected {len(expected)} images, found {len(meta)}")

    # case_id, image_id, split을 기준으로 CSV 순서에 맞게 임베딩 정렬
    meta["_pos"] = np.arange(len(meta))
    matched = expected.merge(meta, on=KEYS, how="left", validate="one_to_one", sort=False)
    if matched["_pos"].isna().any():
        examples = matched.loc[matched["_pos"].isna(), KEYS].head(3).to_dict("records")
        raise ValueError(f"{path}: missing matching embeddings, examples: {examples}")
    return embeddings[matched["_pos"].to_numpy(dtype=np.int64)]


# ============================================================
# 2. 조건별 Test 데이터 평가
# ============================================================
def evaluate_condition(model, embeddings, targets, weights, case_ids, batch_size, device, num_classes):
    # 임베딩 기반 Test DataLoader 생성
    dataset = EmbeddingDataset(embeddings, targets, weights, case_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    records = []
    model.eval()

    # 이미지별 예측 확률 및 정답 레이블 수집
    with torch.inference_mode():
        for batch in loader:
            probs = torch.softmax(model(batch["image"].to(device)), dim=1).cpu().numpy()
            labels = batch["target"].argmax(dim=1).cpu().tolist()
            records.extend((cid, p, int(label)) for cid, p, label in zip(batch["case_id"], probs, labels))

    # 기존 프로젝트의 case 단위 평가 함수 재사용
    return case_metrics(records, num_classes=num_classes, evaluation="test")


# ============================================================
# 3. 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--original-embeddings", type=Path, required=True)
    parser.add_argument("--lowres-embeddings", type=Path, required=True)
    parser.add_argument("--blur-embeddings", type=Path, required=True)
    parser.add_argument("--dark-embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-class-map", type=Path, default=Path("config/class_to_idx.json"))
    parser.add_argument("--class-map", type=Path, default=Path("config/class_to_idx_top11.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/derm_robustness"))
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    # 3-1. 기존 33개 및 신규 11개 클래스 매핑 불러오기
    old_mapping = load_class_to_idx(args.original_class_map)
    new_mapping = load_class_to_idx(args.class_map)

    if len(old_mapping) != 33 or len(new_mapping) != 11 or new_mapping.get("other") != 10:
        raise ValueError("Expected 33 original classes and 11 new classes (other=10)")

    # 3-2. 전체 데이터 CSV 불러오기 및 검증
    df = pd.read_csv(args.data_csv, dtype={"case_id": str, "image_id": str})
    required = set(KEYS) | {
        "soft_label", "sample_weight", "dermatologist_skin_condition_on_label_name",
        "dermatologist_skin_condition_confidence",
    }

    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")
    if df[list(required)].isna().any().any():
        raise ValueError("CSV contains missing required values")
    if not set(df["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Unexpected split value")

    # 3-3. 학습과 동일한 기준으로 레이블 필터링 및 11개 클래스 변환
    old_targets = get_soft_targets(df, old_mapping)
    keep, targets = filter_and_remap(df, old_targets, old_mapping, new_mapping)

    filtered_df = df.loc[keep].reset_index(drop=True)
    weights = pd.to_numeric(filtered_df["sample_weight"], errors="raise").to_numpy(dtype=np.float32)

    if not np.isfinite(weights).all() or ((weights <= 0) | (weights > 1)).any():
        raise ValueError("Invalid sample weights")

    # 3-4. 필터링된 데이터에서 Test 데이터만 선택
    test_mask = filtered_df["split"].eq("test").to_numpy()
    test_df = filtered_df.loc[test_mask].reset_index(drop=True)
    test_targets = targets[test_mask]
    test_weights = weights[test_mask]

    if test_df.empty:
        raise ValueError("No eligible Test images after filtering")

    # 변형 임베딩은 11개 클래스 필터링 이전의 전체 Test 이미지로 추출됨
    all_test_df = df.loc[df["split"].eq("test")].reset_index(drop=True)

    # 전체 Test 이미지 중 실제 평가 대상에 해당하는 행만 선택
    test_keep = np.asarray(keep, dtype=bool)[df["split"].eq("test").to_numpy()]
    assert int(test_keep.sum()) == len(test_df)

    # 3-5. 학습된 DermClassifier 체크포인트 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    if checkpoint.get("class_map") != new_mapping:
        raise ValueError("Checkpoint class_map does not match supplied class map")

    model = DermClassifier(input_dim=6144, num_classes=len(new_mapping)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 3-6. 조건별 임베딩 파일 경로 설정
    paths = {
        "original": args.original_embeddings,
        "lowres": args.lowres_embeddings,
        "blur": args.blur_embeddings,
        "dark": args.dark_embeddings,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Test: {test_df['case_id'].nunique()} cases / {len(test_df)} images")

    rows = []
    original_f1 = None
    original_cases = None
    original_macro_labels = None

    # 3-7. 원본 및 품질 저하 조건별 성능 평가
    for condition in CONDITIONS:
        print(f"\n===== {condition.upper()} =====")

        if condition == "original":
            # 원본 임베딩은 Train/Validation/Test 전체 데이터를 포함
            embeddings = load_aligned_embeddings(paths[condition], df)

            # 학습과 동일한 레이블 필터링 후 Test 데이터만 선택
            embeddings = embeddings[np.asarray(keep, dtype=bool)][test_mask]
        else:
            # 변형 임베딩은 필터링 이전의 전체 Test 데이터를 포함
            embeddings = load_aligned_embeddings(paths[condition], all_test_df)

            # 원본 평가와 동일한 11개 클래스 평가 대상만 선택
            embeddings = embeddings[test_keep]

        if len(embeddings) != len(test_df):
            raise ValueError(f"{condition}: Test embedding count mismatch")

        # 동일한 분류기를 사용하여 조건별 Test 성능 계산
        metrics = evaluate_condition(
            model, embeddings, test_targets, test_weights,
            test_df["case_id"].tolist(), args.batch_size, device, len(new_mapping),
        )

        # 원본 이미지의 성능 및 평가 대상 클래스 정보 저장
        if condition == "original":
            original_f1 = float(metrics["macro_f1"])
            original_cases = int(metrics["cases"])
            original_macro_labels = list(metrics["macro_labels"])

        # 변형 이미지도 원본과 동일한 케이스 및 클래스 기준으로 평가되었는지 검증
        elif int(metrics["cases"]) != original_cases or list(metrics["macro_labels"]) != original_macro_labels:
            raise ValueError(f"{condition}: evaluated cases or Macro F1 classes changed")

        # 3-8. 원본 대비 Macro F1 감소폭 및 강건성 기준 충족 여부 계산
        drop = original_f1 - float(metrics["macro_f1"])

        rows.append({
            "condition": condition,
            "cases": int(metrics["cases"]),
            "images": len(embeddings),
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "macro_num_classes": int(metrics["macro_num_classes"]),
            "macro_f1_drop": drop,
            "pass_10pp": bool(drop <= 0.10 + 1e-12),
        })

        # 조건별 클래스 성능 결과 CSV 저장
        build_class_report(metrics, new_mapping).to_csv(
            args.output_dir / f"derm_{condition}_classwise.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"Cases: {metrics['cases']} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"Drop: {drop:.4f}"
        )

    # 3-9. 조건별 강건성 평가 결과 저장
    summary = pd.DataFrame(rows)
    output_path = args.output_dir / "derm_robustness_results.csv"

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== ROBUSTNESS SUMMARY =====")
    print(summary.to_string(index=False))

    # 3-10. 모든 품질 저하 조건에서 Macro F1 감소폭 10%p 이하인지 확인
    passed = bool(summary.loc[summary["condition"] != "original", "pass_10pp"].all())

    print("\nRequirement (Macro F1 drop <= 10%p):", "PASS" if passed else "FAIL")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
