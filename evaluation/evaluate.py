
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import ImageClassifier, MultimodalClassifier
from training.train import SCINDataset, run_epoch, case_metrics


# ============================================================
# 클래스별 Precision / Recall / F1 / Case 수 표 생성
# ============================================================
def build_class_report(metrics, mapping):

    idx_to_class = {
        idx: name
        for name, idx in mapping.items()
    }

    rows = []

    for idx in range(len(mapping)):
        support = int(metrics["class_support"][idx])

        rows.append({
            "class_index": idx,
            "class_name": idx_to_class[idx],

            "precision": (
                float(metrics["class_precision"][idx])
                if support > 0 else None
            ),

            "recall": (
                float(metrics["class_recall"][idx])
                if support > 0 else None
            ),

            "f1": (
                float(metrics["class_f1"][idx])
                if support > 0 else None
            ),

            "cases": support,

            "status": (
                "Available"
                if support > 0
                else "No test samples"
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# 특정 이미지 조건에서 모델 평가
# original / lowres / blur / dark
# ============================================================
def evaluate_condition(
    model,
    test_df,
    condition_dir,
    mapping,
    symptom_cols,
    batch_size,
    device
):

    df = test_df.copy()

    # 기존 image_path의 파일명만 유지하고
    # condition 폴더의 이미지로 변경
    df["image_path"] = df["image_path"].apply(
        lambda x: str(
            condition_dir / Path(str(x)).name
        )
    )

    # 파일 존재 확인
    missing_files = [
        p
        for p in df["image_path"]
        if not Path(p).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} test images are missing "
            f"in {condition_dir}"
        )

    dataset = SCINDataset(
        df,
        root=Path("."),
        mapping=mapping,
        symptom_cols=symptom_cols,
        train=False
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loss, records = run_epoch(
        model,
        loader,
        device
    )

    # Test는 실제 정답에 존재하는 클래스만 Macro F1 계산
    metrics = case_metrics(
        records,
        num_classes=len(mapping),
        evaluation="test"
    )

    return test_loss, metrics


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="final_integrated_dataset.csv 경로"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="학습된 모델 checkpoint 경로"
    )

    parser.add_argument(
        "--robustness-root",
        type=Path,
        required=True,
        help="original/lowres/blur/dark 폴더가 들어있는 test_images 경로"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results")
    )

    args = parser.parse_args()


    # ========================================================
    # 1. 학습된 checkpoint 불러오기
    # ========================================================
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    state = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True
    )

    mapping = state["class_map"]
    symptom_cols = state["symptom_cols"]
    mode = state["mode"]


    # ========================================================
    # 2. Test 데이터 불러오기
    # ========================================================
    test_df = pd.read_csv(
        args.test_csv,
        dtype={"case_id": str}
    )

    # 통합 CSV인 경우 test split만 선택
    if "split" in test_df.columns:
        test_df = test_df[
            test_df["split"] == "test"
        ].copy()

    if test_df.empty:
        raise ValueError(
            "Test 데이터가 없습니다."
        )


    required = {
        "case_id",
        "image_path",
        "soft_label",
        "sample_weight",
        *symptom_cols
    }

    missing = (
        required
        - set(test_df.columns)
    )

    if missing:
        raise ValueError(
            f"Test CSV missing columns: {missing}"
        )

    if test_df[
        list(required)
    ].isna().any().any():
        raise ValueError(
            "Test CSV contains missing required values"
        )


    # ========================================================
    # 3. 모델 생성
    # ========================================================
    if mode == "image":

        model = ImageClassifier(
            num_classes=len(mapping),
            pretrained=False
        )

    elif mode == "multimodal":

        model = MultimodalClassifier(
            num_classes=len(mapping),
            symptom_dim=len(symptom_cols),
            pretrained=False,
            freeze_backbone=True
        )

    else:
        raise ValueError(
            f"Unknown model mode: {mode}"
        )


    model.load_state_dict(
        state["model"]
    )

    model = model.to(device)


    # ========================================================
    # 4. Original / Lowres / Blur / Dark 평가
    # ========================================================
    conditions = [
        "original",
        "lowres",
        "blur",
        "dark"
    ]

    robustness_rows = []
    class_reports = []

    for condition in conditions:

        condition_dir = (
            args.robustness_root
            / condition
        )

        if not condition_dir.exists():
            raise FileNotFoundError(
                f"Folder not found: {condition_dir}"
            )

        print(
            f"\n===== {condition.upper()} ====="
        )

        test_loss, metrics = evaluate_condition(
            model=model,
            test_df=test_df,
            condition_dir=condition_dir,
            mapping=mapping,
            symptom_cols=symptom_cols,
            batch_size=args.batch_size,
            device=device
        )


        # ----------------------------
        # 전체 성능 저장
        # ----------------------------
        robustness_rows.append({
            "model": mode,
            "condition": condition,

            "cases": metrics["cases"],
            "loss": test_loss,

            "accuracy":
                metrics["accuracy"],

            "macro_f1":
                metrics["macro_f1"],

            "macro_num_classes":
                metrics["macro_num_classes"]
        })


        # ----------------------------
        # 클래스별 성능 저장
        # ----------------------------
        class_report = build_class_report(
            metrics,
            mapping
        )

        class_report.insert(
            0,
            "condition",
            condition
        )

        class_report.insert(
            0,
            "model",
            mode
        )

        class_reports.append(
            class_report
        )


        print(
            f"Cases: {metrics['cases']}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Macro F1 classes: "
            f"{metrics['macro_num_classes']}"
        )


    # ========================================================
    # 5. 결과 DataFrame 생성
    # ========================================================
    robustness_df = pd.DataFrame(
        robustness_rows
    )

    classwise_df = pd.concat(
        class_reports,
        ignore_index=True
    )


    # ========================================================
    # 6. Original 대비 성능 감소량 계산
    # ========================================================
    original_f1 = robustness_df.loc[
        robustness_df["condition"]
        == "original",
        "macro_f1"
    ].iloc[0]

    robustness_df[
        "macro_f1_drop"
    ] = (
        original_f1
        - robustness_df["macro_f1"]
    )


    # ========================================================
    # 7. 결과 출력
    # ========================================================
    print(
        "\n===== ROBUSTNESS RESULTS ====="
    )

    print(
        robustness_df[
            [
                "condition",
                "cases",
                "accuracy",
                "macro_f1",
                "macro_f1_drop"
            ]
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # 8. CSV 저장
    # ========================================================
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_name = args.checkpoint.stem

    robustness_path = (
        args.output_dir
        / f"{checkpoint_name}_robustness_results.csv"
    )

    classwise_path = (
        args.output_dir
        / f"{checkpoint_name}_classwise_results.csv"
    )

    robustness_df.to_csv(
        robustness_path,
        index=False,
        encoding="utf-8-sig"
    )

    classwise_df.to_csv(
        classwise_path,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "\n===== SAVED ====="
    )

    print(
        f"Robustness results: "
        f"{robustness_path}"
    )

    print(
        f"Class-wise results: "
        f"{classwise_path}"
    )


if __name__ == "__main__":
    main()
