import pandas as pd
import numpy as np
import ast
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"

cases_path = DATASET_DIR / "scin_cases_with_split.csv"
images_path = DATASET_DIR / "scin_image_samples.csv"

cases_df = pd.read_csv(cases_path, dtype={"case_id": "string"})
images_df = pd.read_csv(images_path, dtype={"case_id": "string"})


# 선정한 32개 클래스
CLASSES = [
    "Eczema",
    "Allergic Contact Dermatitis",
    "Urticaria",
    "Insect Bite",
    "Folliculitis",
    "Psoriasis",
    "Tinea",
    "Herpes Simplex",
    "Acne",
    "CD - Contact dermatitis",
    "Herpes Zoster",
    "Irritant Contact Dermatitis",
    "Pigmented purpuric eruption",
    "Tinea Versicolor",
    "Pityriasis rosea",
    "Acute dermatitis, NOS",
    "Impetigo",
    "Keratosis pilaris",
    "O/E - ecchymoses present",
    "Lichen Simplex Chronicus",
    "Rosacea",
    "Leukocytoclastic Vasculitis",
    "Granuloma annulare",
    "Drug Rash",
    "Prurigo nodularis",
    "Verruca vulgaris",
    "Lichen planus/lichenoid eruption",
    "Molluscum Contagiosum",
    "Acute and chronic dermatitis",
    "Viral Exanthem",
    "Purpura",
    "Hypersensitivity"
]

ALL_CLASSES = CLASSES + ["other"]


# 클래스별 인덱스 매핑
class_to_idx = {
    name: idx for idx, name in enumerate(ALL_CLASSES)
}


# ============================================================
# Soft Label 생성
# ============================================================
def make_soft_label(value):
    # 입력 예시: "{'Eczema': 0.7, 'Acne': 0.3}" -> 딕셔너리로 변환
    labels = ast.literal_eval(value) if isinstance(value, str) else value

    # 33개 클래스의 확률값을 모두 0으로 초기화
    soft_label = np.zeros(len(ALL_CLASSES), dtype=np.float32)

    for disease, weight in labels.items():
        idx = class_to_idx.get(disease, class_to_idx["other"])
        soft_label[idx] += float(weight)

    total = soft_label.sum()

    if total <= 0:
        raise ValueError(f"유효하지 않은 레이블: {value}")

    # 가중치 총합이 1이 되도록 정규화
    soft_label /= total

    return soft_label.tolist()


# ============================================================
# Sample Weight 계산
# ============================================================
def make_sample_weight(value):
    confidences = ast.literal_eval(value) if isinstance(value, str) else value

    valid = [
        float(c) for c in confidences
        if c is not None and not pd.isna(c)
    ]

    if not valid:
        raise ValueError(f"유효한 confidence가 없습니다: {value}")

    # confidence 최댓값을 5로 나누어 sample weight 계산
    return max(valid) / 5.0


# ============================================================
# 1. 케이스별 레이블 전처리
# ============================================================
labels_df = cases_df[["case_id", "split"]].copy()

# 각 케이스의 질환별 가중치를 33차원 soft label로 변환
labels_df["soft_label"] = cases_df[
    "weighted_skin_condition_label"
].apply(make_soft_label)

# 신뢰도를 기반으로 각 케이스의 sample weight 계산
labels_df["sample_weight"] = cases_df[
    "dermatologist_skin_condition_confidence"
].apply(make_sample_weight)


# ============================================================
# 2. 이미지별 샘플에 레이블 연결
# ============================================================
# 이미지 단위 데이터와 케이스 단위 레이블 데이터를 case_id로 연결
image_labels_df = images_df.merge(
    labels_df.drop(columns=["split"]),
    on="case_id",
    how="left",
    validate="many_to_one"
)


# ============================================================
# 3. 전처리 결과 검증
# ============================================================
assert len(labels_df) == 1850
assert labels_df["case_id"].is_unique
assert set(image_labels_df["case_id"]) == set(labels_df["case_id"])
assert len(image_labels_df) == len(images_df)
assert image_labels_df["soft_label"].notna().all()
assert image_labels_df["sample_weight"].notna().all()

assert all(
    len(label) == 33 and np.isclose(sum(label), 1.0)
    for label in labels_df["soft_label"]
)

assert labels_df["sample_weight"].isin([0.6, 0.8, 1.0]).all()

split_map = labels_df.set_index("case_id")["split"]
assert (
    image_labels_df["split"]
    == image_labels_df["case_id"].map(split_map)
).all()


print("케이스별 레이블 수:", len(labels_df))
print("이미지별 레이블 수:", len(image_labels_df))
print("Soft Label 차원:", len(labels_df.iloc[0]["soft_label"]))
print("Sample Weight 범위:",
      labels_df["sample_weight"].min(),
      labels_df["sample_weight"].max())


# ============================================================
# 4. 전처리 결과 저장
# ============================================================
OUTPUT_DIR = DATASET_DIR / "preprocessed_labels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 리스트를 JSON 문자열로 저장
labels_df["soft_label"] = labels_df["soft_label"].apply(json.dumps)
image_labels_df["soft_label"] = image_labels_df["soft_label"].apply(json.dumps)

labels_df.to_csv(OUTPUT_DIR / "scin_case_labels.csv", index=False)
image_labels_df.to_csv(OUTPUT_DIR / "scin_image_labels.csv", index=False)

CONFIG_DIR = BASE_DIR / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_DIR / "class_to_idx.json", "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

print("\nscin_case_labels.csv, scin_image_labels.csv, class_to_idx.json 저장 완료")