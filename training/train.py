"""
SCIN 이미지 단독 / 멀티모달 모델 학습 파이프라인

- Train / Validation 데이터 로딩
- Soft Label 및 Sample Weight 기반 학습
- 케이스 단위 성능 검증
- Validation Loss 기반 최적 모델 저장 및 Early Stopping
"""

# 실행 옵션(--epochs, --mode 등)을 명령줄에서 전달받기 위한 모듈
from tqdm.auto import tqdm
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models import ImageClassifier, MultimodalClassifier


# 실험을 재현하기 위해 Python, NumPy, PyTorch의 난수 시드 고정
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_to_idx(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    mapping = {str(k): int(v) for k, v in raw.items()}
    assert sorted(mapping.values()) == list(range(len(mapping))), 'Class indices must be contiguous from zero'
    assert 'other' in mapping, 'Class mapping must include other'
    return mapping


# JSON 형태의 Soft Label을 클래스 순서에 맞는 확률 벡터로 변환
# 예: {'Eczema': 0.7, 'other': 0.3} -> 길이 33의 텐서
def parse_soft(value, mapping):
    obj = json.loads(value) if isinstance(value, str) else value
    if isinstance(obj, dict):
        arr = np.zeros(len(mapping), dtype=np.float32)
        for key, weight in obj.items():
            if key not in mapping:
                raise ValueError(f'Unknown label: {key}')
            arr[mapping[key]] = float(weight)
    else:
        arr = np.asarray(obj, dtype=np.float32)
    if arr.shape != (len(mapping),) or not np.isfinite(arr).all() or (arr < 0).any() or arr.sum() <= 0:
        raise ValueError('Invalid soft label')
    return torch.tensor(arr / arr.sum(), dtype=torch.float32)


# ========================================
# 전처리된 데이터 조회 -> 이미지, 증상, 레이블, 가중치를 반환하는 Dataset 클래스
# ========================================
class SCINDataset(Dataset):
    # frame: 해당 split의 CSV, root: 이미지 파일 기준 폴더
    def __init__(self, frame, root, mapping, symptom_cols, train=False):
        self.df = frame.reset_index(drop=True)  # 행 인덱스를 0부터 다시 부여
        self.root = Path(root)
        self.mapping = mapping
        self.symptom_cols = symptom_cols
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224))] +
            (
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                ] if train else [] # Train Augmentation
            ) + [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ]
            )

    def __len__(self):
        return len(self.df)

    # 인덱스에 해당하는 샘플 조회 -> 이미지 로드 및 전처리
    # -> 증상 데이터 Tensor 변환 -> 이미지, 증상, Soft Label, Sample Weight 반환
    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_id = str(row['image_id'])
        split = str(row['split'])

        if split == 'train':
            path = self.root / 'train_images' / 'original' / f'{image_id}.jpg'
        elif split == 'validation':
            path = self.root / 'validation_images' / f'{image_id}.jpg'
        else:
            raise ValueError(f'Unsupported split: {split}')

        with Image.open(path) as im:
            image = self.transform(im.convert('RGB'))

        symptoms = torch.tensor(
            row[self.symptom_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32
            )
        
        weight = float(row['sample_weight'])
        if not (0 < weight <= 1):
            raise ValueError('sample_weight must be in (0,1]')
        return {'image': image, 'symptoms': symptoms,
                'target': parse_soft(row['soft_label'], self.mapping),
                'weight': torch.tensor(weight, dtype=torch.float32),
                'case_id': str(row['case_id'])}


# 클래스 가중치 계산
def compute_class_weights(train_df, mapping):
    soft_labels = np.stack([
        parse_soft(value, mapping).numpy()
        for value in train_df["soft_label"]
    ])

    sample_weights = train_df["sample_weight"].to_numpy(
        dtype=np.float32
    )

    # 실제 이미지 단위 학습 기준
    class_mass = (
        soft_labels * sample_weights[:, None]
    ).sum(axis=0)

    class_weights = 1.0 / np.sqrt(class_mass + 1e-8)
    class_weights = class_weights / class_weights.mean()
    class_weights = np.clip(class_weights, 0.5, 3.0)

    return torch.tensor(class_weights, dtype=torch.float32)


# Soft Label과 Sample weight를 반영한 Cross Entropy
def weighted_soft_ce(logits,target, weights,class_weights=None):
    log_probs = torch.log_softmax(logits, dim=1)

    if class_weights is not None:
        per_sample = -(
            target * log_probs * class_weights.unsqueeze(0)
        ).sum(dim=1)
    else:
        per_sample = -(target * log_probs).sum(dim=1)

    return (
        (per_sample * weights).sum()
        / weights.sum().clamp_min(1e-8)
    )


# 한 Epoch 동안 학습 또는 검증 수행
def run_epoch(model, loader, device, optimizer=None, class_weights=None):
    # Optimizer 존재 여부로 학습/검증 모드 구분
    training = optimizer is not None
    model.train(training)

    total_numerator, total_denominator = 0., 0.
    records = []

    for batch in tqdm(
      loader,
      desc="Train" if training else "Validation",
      unit="batch"
    ):
        image = batch['image'].to(device)
        symptoms = batch['symptoms'].to(device)
        target = batch['target'].to(device)
        weight = batch['weight'].to(device)

        # 모델 예측 및 Soft Label 기반 Loss 계산
        with torch.set_grad_enabled(training):
            logits = model(image, symptoms)
            loss = weighted_soft_ce(logits, target, weight, class_weights=class_weights)

            if training:  # 학습 모드일 때만 역전파 및 가중치 업데이트
                optimizer.zero_grad()  
                loss.backward()
                optimizer.step()

        # 전체 Epoch의 가중 평균 Loss 계산을 위한 누적
        total_numerator += loss.item() * weight.sum().item()
        total_denominator += weight.sum().item()

        # 검증 모드에서는 케이스별 평가를 위한 예측 결과 기록
        if not training:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            for cid, p, t in zip(
                batch['case_id'],
                probs,
                target.argmax(dim=1).cpu().tolist()
            ):
                records.append((cid, p, t))
    return total_numerator / max(total_denominator, 1e-8), records

# ========================================
# 동일 케이스의 이미지별 예측 확률을 평균 내어 최종 예측을 결정
# 케이스 단위 Accuracy와 Macro F1을 계산
# ========================================
def case_metrics(records, num_classes=33, evaluation="validation"):
    """
    케이스 단위 성능 평가

    validation:
        전체 33개 클래스 기준 Macro F1

    test:
        실제 Test 정답에 존재하는 클래스만 기준 Macro F1
    """
    grouped = {}

    # 동일 case의 이미지 예측 결과 그룹화
    for cid, prob, label in records:
        if cid in grouped:
            if grouped[cid]["label"] != label:
                raise ValueError(f'Inconsistent targets for case {cid}')
            grouped[cid]["probs"].append(prob)
        else:
            grouped[cid] = {
                "probs": [prob],
                "label": label
            }

    y_true = []
    y_pred = []
    case_probs = {}

    # 같은 case의 여러 이미지 확률 평균
    for cid, item in grouped.items():
        mean_prob = np.mean(item["probs"], axis=0)

        true_label = int(item["label"])
        pred_label = int(mean_prob.argmax())

        y_true.append(true_label)
        y_pred.append(pred_label)
        case_probs[cid] = mean_prob

    # Macro F1 계산 기준
    if evaluation == "validation":
        macro_labels = list(range(num_classes))

    elif evaluation == "test":
        macro_labels = sorted(set(y_true))

    else:
        raise ValueError(
            "evaluation must be 'validation' or 'test'"
        )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=macro_labels,
        average="macro",
        zero_division=0
    )

    # 클래스별 성능은 전체 33개 출력
    all_labels = list(range(num_classes))

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=all_labels,
            zero_division=0
        )
    )

    return {
        "cases": len(grouped),
        "accuracy": accuracy,
        "macro_f1": macro_f1,

        "macro_labels": macro_labels,
        "macro_num_classes": len(macro_labels),

        "class_precision": precision,
        "class_recall": recall,
        "class_f1": f1,
        "class_support": support,

        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "case_probs": case_probs
    }


# ========================================
# 데이터 로딩 및 검증 -> 모델 초기화 -> 학습/검증 -> 최적 모델 저장
# ========================================
def main():
    # 1. 실행 인자 설정 (하이퍼파라미터 및 파일 경로)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=Path, required=True, help='split 컬럼이 포함된 CSV 파일 경로')
    parser.add_argument('--image-root', type=Path, default=Path('.'))
    parser.add_argument('--class-map', type=Path, default=Path('config/class_to_idx.json'))
    parser.add_argument('--mode', choices=['image', 'multimodal'], default='image')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=Path('outputs'))
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--class-weighting', action='store_true')
    args = parser.parse_args()

    # 2. 실험 시드를 고정 및 클래스 인덱스 불러오기
    seed_everything(args.seed)
    mapping = load_class_to_idx(args.class_map)

    # 3. 데이터 불러오기
    df = pd.read_csv(
      args.data_csv,
      dtype={'case_id': str, 'image_id': str},
      keep_default_na=True
      )

    # Train / Validation 분리
    frames = {
      'train': df[df['split'] == 'train'].copy(),
      'val': df[df['split'] == 'validation'].copy()
    }
    
    # 4. 데이터 유효성 검증
    required = {'case_id', 'image_id', 'soft_label', 'sample_weight'}
    for split, frame in frames.items():
        if not required.issubset(frame.columns):
            raise ValueError(f'{split} missing {required - set(frame.columns)}')
        if frame[list(required)].isna().any().any():
            raise ValueError(f'{split} contains missing required values')

    # 4-1. Train / Validation 간 case_id 중복 확인
    sets = {s: set(df.case_id) for s, df in frames.items()}
    assert not (sets['train'] & sets['val']), 'case_id leakage'
    
    
    # 4-2. 증상, 피부 특징, 발생 부위 컬럼 확인
    symptom_cols = sorted(
        c for c in frames['train'].columns
        if c.startswith(('symptoms_', 'textures_', 'body_parts_'))
    )

    for split, frame in frames.items():
        if not set(symptom_cols).issubset(frame.columns):
            raise ValueError(f'{split} missing symptom columns')
        # 4-3. 동일 case_id의 레이블 및 가중치 일관성 확인
        for cid, group in frame.groupby('case_id'):
            if group['soft_label'].nunique() != 1 or group['sample_weight'].nunique() != 1:
                raise ValueError(f'Inconsistent case metadata: {split}/{cid}')
                
    # 5. Dataset 및 DataLoader 생성
    datasets = {
        s: SCINDataset(
            frames[s],
            args.image_root,
            mapping,
            symptom_cols,
            train=(s == 'train')
        )
        for s in frames
    }

    loaders = {
        s: DataLoader(
            datasets[s],
            batch_size=args.batch_size,
            shuffle=(s == 'train'),
            num_workers=0
        )
        for s in datasets
    }

    # 6. 모델 및 학습 환경 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 6-0. 클래스 불균형 보정을 위한 가중치 설정
    class_weights = None
    if args.class_weighting:
        class_weights = compute_class_weights(
            frames["train"],
            mapping
        ).to(device)

    # 6-1. 실행 모드에 따라 이미지 단독 또는 멀티모달 모델 생성
    if args.mode == 'image':
      model = ImageClassifier(
      num_classes=len(mapping),
      pretrained=not args.no_pretrained
      )

    else:
      model = MultimodalClassifier(
      num_classes=len(mapping),
      symptom_dim=len(symptom_cols),
      pretrained=False,
      freeze_backbone=True
      )

    model = model.to(device)


    # 6-2. 멀티모달인 경우 이미지 단독 모델의 Backbone 가중치 불러오기
    if args.mode == 'multimodal':
      state = torch.load(
          args.out / 'image_best.pt',
          map_location='cpu',
          weights_only=True
          )

      backbone_weights = {
          key.removeprefix('backbone.'): value
          for key, value in state['model'].items()
          if key.startswith('backbone.')
      }

      model.backbone.load_state_dict(backbone_weights)

    # 6-3. Optimizer 설정
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
        )

    # 7. 최적 모델 체크포인트 저장 경로 및 Early Stopping 설정
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / f'{args.mode}_best.pt'
    best, stale = float('-inf'), 0

    # 8. 모델 학습 및 검증
    for epoch in range(1, args.epochs + 1):
        # 8-1. Train 데이터로 모델 학습
        train_loss, _ = run_epoch(model, loaders['train'], device, optimizer, class_weights=class_weights)
        # 8-2. Validation 데이터로 성능 검증
        val_loss, val_records = run_epoch(model, loaders['val'], device)
        metrics = case_metrics(
            val_records,
            num_classes=len(mapping),
            evaluation="validation"
        )

        print(
            f'Epoch {epoch}: '
            f'train_loss={train_loss:.4f}, '
            f'val_loss={val_loss:.4f}, '
            f'val_macro_f1={metrics["macro_f1"]:.4f}'
        )

        idx_to_class = {v: k for k, v in mapping.items()}

        # 클래스별 평가 결과
        class_results = pd.DataFrame({
            "class_name": [
                idx_to_class[i] for i in range(len(mapping))
            ],
            "precision": metrics["class_precision"],
            "recall": metrics["class_recall"],
            "f1": metrics["class_f1"],
            "support": metrics["class_support"]
        })

        # F1이 낮은 클래스부터 출력
        print("\n=== 클래스별 성능 ===")
        print(
            class_results
            .sort_values("f1")
            .to_string(index=False)
        )

        # 8-3. 검증 손실이 개선되면 최적 모델 저장
        if metrics['macro_f1'] > best:
            best, stale = metrics['macro_f1'], 0
            torch.save({
                'model': model.state_dict(),
                'class_map': mapping,
                'symptom_cols': symptom_cols,
                'mode': args.mode,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_macro_f1': metrics['macro_f1'],
                'config': {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                    }
                }, checkpoint)
        else: # 검증 손실이 개선되지 않으면 Early Stopping 확인
            stale += 1
            if stale >= args.patience:
                print('Early stopping')
                break

    # 9. 최적 모델 저장 경로 출력
    print(f'Training completed. Best model saved to: {checkpoint}')


if __name__ == '__main__':
    main()