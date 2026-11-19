import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download

EMBEDDING_DIM = 6144

DEFAULT_OUT = Path("/content/drive/MyDrive/데이터사이언스응용/embeddings/derm_embeddings.npz")

# ========================================
# 1. 이미지 경로
# ========================================
def resolve_image_path(root, row):
    image_id = str(row["image_id"])
    split = str(row["split"])

    if split == "train":
        path = root / "train_images" / "original" / f"{image_id}.jpg"
    elif split == "validation":
        path = root / "validation_images" / f"{image_id}.jpg"
    elif split == "test":
        path = root / "test_images" / "original" / f"{image_id}.jpg"
    else:
        raise ValueError(f"Unsupported split: {split}")

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


# ========================================
# 2. 이미지 인코딩
# ========================================
def encode_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((448, 448))

        buffer = BytesIO()
        image.save(buffer, format="PNG")

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[buffer.getvalue()]
                    )
                )
            }
        )
    )

    return example.SerializeToString()


# ========================================
# 3. 전체 이미지 경로 검증
# ========================================
def validate_paths(df, root):
    paths = []
    missing = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Checking paths"
    ):
        try:
            paths.append(resolve_image_path(root, row))
        except FileNotFoundError as e:
            missing.append(str(e))
            paths.append(None)

    if missing:
        print("Missing image examples:")
        for path in missing[:10]:
            print(path)

        raise FileNotFoundError(
            f"{len(missing)} images are missing"
        )

    print(f"All {len(paths)} image paths are valid.")
    return paths


# ========================================
# 4. Derm Foundation 모델 추론
# ========================================
def infer_batch(infer, paths):
    serialized = [
        encode_image(path)
        for path in paths
    ]

    output = infer(
        inputs=tf.constant(
            serialized,
            dtype=tf.string
        )
    )

    embeddings = output["embedding"].numpy()

    if embeddings.size != len(paths) * EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    return embeddings.reshape(
        len(paths), EMBEDDING_DIM
    ).astype(np.float32)


# ========================================
# 5. 진행 상태 저장
# ========================================
def save_progress(
    embeddings,
    meta_path,
    completed,
    case_ids,
    image_ids,
    splits
):
    # 임베딩 데이터를 디스크에 반영
    embeddings.flush()

    # 메타데이터를 임시 파일에 먼저 저장
    temp_path = meta_path.with_name(
        meta_path.name + ".tmp"
    )

    with open(temp_path, "wb") as f:
        np.savez(
            f,
            completed=completed,
            case_ids=case_ids,
            image_ids=image_ids,
            splits=splits
        )

    # 저장 완료 후 기존 메타데이터 교체
    temp_path.replace(meta_path)

    print(f"Progress saved: {completed}/{len(case_ids)}")


# ========================================
# 6. 메인
# ========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    if args.batch_size < 1 or args.save_every < 1:
        raise ValueError(
            "batch-size and save-every must be positive"
        )

    # ========================================
    # 6-1. CSV 불러오기
    # ========================================
    df = pd.read_csv(
        args.data_csv,
        dtype={"case_id": str, "image_id": str}
    )

    required_columns = {"case_id", "image_id", "split"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {required_columns - set(df.columns)}"
        )

    if df[list(required_columns)].isna().any().any():
        raise ValueError(
            "CSV contains missing case_id, image_id or split"
        )

    case_ids = df["case_id"].to_numpy(dtype=str)
    image_ids = df["image_id"].to_numpy(dtype=str)
    splits = df["split"].astype(str).to_numpy()

    # ========================================
    # 6-2. 이미지 경로 검증
    # ========================================
    paths = validate_paths(df, args.image_root)

    # ========================================
    # 6-3. 저장 경로 설정
    # ========================================
    args.out.parent.mkdir(parents=True, exist_ok=True)

    progress_path = args.out.with_suffix(".progress.npy")
    meta_path = args.out.with_suffix(".progress_meta.npz")

    print("Output directory:", args.out.parent)

    # ========================================
    # 6-4. 기존 작업 재개
    # ========================================
    if progress_path.exists() and meta_path.exists():

        with np.load(meta_path, allow_pickle=True) as meta:
            if (
                not np.array_equal(meta["case_ids"], case_ids)
                or not np.array_equal(meta["image_ids"], image_ids)
                or not np.array_equal(meta["splits"], splits)
            ):
                raise ValueError(
                    "CSV differs from saved progress. "
                    "Use a new output path."
                )

            start = int(meta["completed"])

        embeddings = np.load(
            progress_path,
            mmap_mode="r+"
        )

        if embeddings.shape != (len(df), EMBEDDING_DIM):
            raise ValueError("Progress shape mismatch")

        if not 0 <= start <= len(df):
            raise ValueError(f"Invalid progress position: {start}")

        print(f"Resuming from {start}/{len(df)}")

    else:
        if progress_path.exists() or meta_path.exists():
            raise RuntimeError(
                "Incomplete progress files found. "
                "Check or remove them before restarting."
            )

        start = 0

        embeddings = np.lib.format.open_memmap(
            progress_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(df), EMBEDDING_DIM)
        )

        save_progress(
            embeddings,
            meta_path,
            0,
            case_ids,
            image_ids,
            splits
        )

    # ========================================
    # 6-5. Derm Foundation 모델 로딩
    # ========================================
    if start < len(df):
        print("Downloading/loading Derm Foundation...")

        model_path = snapshot_download(
            repo_id="google/derm-foundation"
        )

        model = tf.saved_model.load(model_path)
        infer = model.signatures["serving_default"]

        # ========================================
        # 6-6. 배치 추론
        # ========================================
        batch_size = args.batch_size
        last_saved = start

        with tqdm(
            total=len(df),
            initial=start,
            desc="Extracting embeddings"
        ) as pbar:

            i = start

            while i < len(df):
                end = min(i + batch_size, len(df))

                current_paths = paths[i:end]

                try:
                    batch_embeddings = infer_batch(
                        infer,
                        current_paths
                    )

                except (tf.errors.OpError, ValueError) as e:
                    if batch_size == 1:
                        raise

                    print(
                        "\nBatch inference failed. "
                        "Switching to single-image inference."
                    )
                    print(str(e)[:500])

                    batch_size = 1
                    continue

                embeddings[i:end] = batch_embeddings

                pbar.update(end - i)
                i = end

                # 마지막 저장 이후 지정한 이미지 수만큼
                # 처리했거나 전체 추출이 끝나면 저장
                if (
                    i - last_saved >= args.save_every
                    or i == len(df)
                ):
                    save_progress(embeddings, meta_path, i, case_ids, image_ids, splits)
                    last_saved = i

    else:
        print("All embeddings already extracted.")

    # ========================================
    # 6-7. 최종 NPZ 저장
    # ========================================
    print("Saving final embeddings...")

    embeddings.flush()

    np.savez_compressed(
        args.out,
        embeddings=np.asarray(embeddings),
        row_indices=np.arange(len(df)),
        case_ids=case_ids,
        image_ids=image_ids,
        splits=splits
    )

    print("Saved:", args.out)
    print("Shape:", embeddings.shape)
    print("Extraction completed successfully.")


if __name__ == "__main__":
    main()
