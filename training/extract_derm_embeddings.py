
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
# 3. 전체 경로 검증
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
# 4. 모델 추론
# ========================================
def infer_batch(infer, paths):
    serialized = [encode_image(path) for path in paths]

    output = infer(
        inputs=tf.constant(serialized, dtype=tf.string)
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
# 5. 메인
# ========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("datasets/derm_embeddings.npz"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    if args.batch_size < 1 or args.save_every < 1:
        raise ValueError(
            "batch-size and save-every must be positive"
        )

    df = pd.read_csv(
        args.data_csv,
        dtype={"case_id": str, "image_id": str}
    )

    # 1. 전체 경로 검증
    paths = validate_paths(df, args.image_root)

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 임시 저장 파일
    progress_path = args.out.with_suffix(".progress.npy")
    meta_path = args.out.with_suffix(".progress_meta.npz")

    case_ids = df["case_id"].to_numpy(dtype=str)
    image_ids = df["image_id"].to_numpy(dtype=str)

    # 2. 기존 작업 재개 여부 확인
    if progress_path.exists() and meta_path.exists():
        meta = np.load(meta_path)

        if (
            not np.array_equal(meta["case_ids"], case_ids)
            or not np.array_equal(meta["image_ids"], image_ids)
            or not np.array_equal(
                meta["splits"],
                df["split"].astype(str).to_numpy()
            )
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

    def save_progress(completed):
        embeddings.flush()

        np.savez(
            meta_path,
            completed=completed,
            case_ids=case_ids,
            image_ids=image_ids,
            splits=df["split"].astype(str).to_numpy()
        )

    if start == 0:
        save_progress(0)

    # 3. 모델 로딩
    print("Downloading/loading Derm Foundation...")

    model_path = snapshot_download(
        repo_id="google/derm-foundation"
    )

    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]

    # 4. 배치 추론
    batch_size = args.batch_size

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
                    infer, current_paths
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

            # 중간 저장
            if (
                i % args.save_every == 0
                or i == len(df)
            ):
                save_progress(i)

    # 5. 최종 NPZ 저장
    print("Saving final embeddings...")

    np.savez_compressed(
        args.out,
        embeddings=np.asarray(embeddings),
        row_indices=np.arange(len(df)),
        case_ids=case_ids,
        image_ids=image_ids
    )

    print("Saved:", args.out)
    print("Shape:", embeddings.shape)


if __name__ == "__main__":
    main()
