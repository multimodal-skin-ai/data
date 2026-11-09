import argparse
from pathlib import Path

import pandas as pd


def load_results(results_dir):
    """
    evaluate.py가 생성한 robustness 결과 CSV들을 모두 읽어서 합친다.
    파일명 예:
    image_seed42_best_robustness_results.csv
    multimodal_seed42_best_robustness_results.csv
    """

    files = sorted(
        results_dir.glob("*_robustness_results.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No robustness result CSVs found in {results_dir}"
        )

    frames = []

    for file in files:
        df = pd.read_csv(file)

        filename = file.stem

        # seed 추출
        seed = None

        for part in filename.split("_"):
            if part.startswith("seed"):
                try:
                    seed = int(
                        part.replace("seed", "")
                    )
                except ValueError:
                    pass

        if seed is None:
            raise ValueError(
                f"Seed not found in filename: {file.name}"
            )

        df["seed"] = seed
        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True
    )


def summarize_seed_results(result_df):
    """
    모델 / 조건별 5개 seed 평균 및 표준편차 계산
    """

    required = {
        "model",
        "seed",
        "condition",
        "accuracy",
        "macro_f1"
    }

    missing = required - set(result_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    summary = (
        result_df
        .groupby(
            ["model", "condition"],
            as_index=False
        )
        .agg(
            n_seeds=("seed", "nunique"),

            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),

            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std")
        )
    )

    summary["accuracy_mean_std"] = summary.apply(
        lambda row:
        f'{row["accuracy_mean"]:.4f} ± '
        f'{row["accuracy_std"]:.4f}',
        axis=1
    )

    summary["macro_f1_mean_std"] = summary.apply(
        lambda row:
        f'{row["macro_f1_mean"]:.4f} ± '
        f'{row["macro_f1_std"]:.4f}',
        axis=1
    )

    return summary


def build_model_comparison(summary_df):
    """
    Original Test 기준
    이미지 단독 vs 멀티모달 비교표
    """

    original = summary_df[
        summary_df["condition"] == "original"
    ].copy()

    return original[
        [
            "model",
            "n_seeds",
            "accuracy_mean_std",
            "macro_f1_mean_std"
        ]
    ]


def build_robustness_comparison(summary_df):
    """
    Original / Lowres / Blur / Dark
    강건성 비교표
    """

    return summary_df[
        [
            "model",
            "condition",
            "n_seeds",
            "macro_f1_mean",
            "macro_f1_std",
            "macro_f1_mean_std"
        ]
    ].copy()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evaluation/results")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/summary")
    )

    args = parser.parse_args()


    # 1. seed별 결과 불러오기
    result_df = load_results(
        args.results_dir
    )


    # 2. 평균 / 표준편차 계산
    summary_df = summarize_seed_results(
        result_df
    )


    # 3. 모델 비교표
    model_comparison = build_model_comparison(
        summary_df
    )


    # 4. 강건성 비교표
    robustness_comparison = (
        build_robustness_comparison(
            summary_df
        )
    )


    # 5. 저장 폴더 생성
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # 6. CSV 저장
    result_df.to_csv(
        args.output_dir
        / "all_seed_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        args.output_dir
        / "seed_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    model_comparison.to_csv(
        args.output_dir
        / "model_test_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    robustness_comparison.to_csv(
        args.output_dir
        / "robustness_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )


    # 7. 출력
    print(
        "\n===== MODEL TEST COMPARISON ====="
    )

    print(
        model_comparison.to_string(
            index=False
        )
    )

    print(
        "\n===== ROBUSTNESS COMPARISON ====="
    )

    print(
        robustness_comparison.to_string(
            index=False
        )
    )

    print(
        "\n✅ 5-seed summary completed!"
    )


if __name__ == "__main__":
    main()