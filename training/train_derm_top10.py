import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from training.train import weighted_soft_ce, seed_everything, load_class_to_idx
from models.backbones.derm_foundation import DermClassifier


# 사전 추출된 임베딩 사용
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets, weights, hard_labels, case_ids):
        self.embeddings = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.targets = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
        self.weights = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32))
        self.hard_labels = torch.from_numpy(np.ascontiguousarray(hard_labels, dtype=np.int64))
        self.case_ids = list(case_ids)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, i):
        return {'image': self.embeddings[i], 'target': self.targets[i], 'weight': self.weights[i], 'hard_label': self.hard_labels[i], 'case_id': self.case_ids[i]}


def load_embeddings(path):
    with np.load(path, allow_pickle=True) as data:
        required = {'embeddings', 'case_ids', 'image_ids'}
        if not required.issubset(data.files):
            raise ValueError(f'필수 배열 누락: {required - set(data.files)}')
        embeddings = np.asarray(data['embeddings'], dtype=np.float32)
        case_ids = data['case_ids'].astype(str)
        image_ids = data['image_ids'].astype(str)
    if not len(embeddings) == len(case_ids) == len(image_ids):
        raise ValueError('임베딩과 식별자 개수가 일치하지 않습니다.')
    if embeddings.ndim != 2 or embeddings.shape[1] != 6144:
        raise ValueError(f'임베딩 차원 오류: {embeddings.shape}')
    if not np.isfinite(embeddings).all():
        raise ValueError('임베딩에 NaN 또는 Inf가 포함되어 있습니다.')

    embedding_map = {}
    for idx, (cid, iid) in enumerate(zip(case_ids, image_ids)):
        key = (cid, iid)
        if key in embedding_map:
            raise ValueError(f'중복 이미지 식별자: {key}')
        embedding_map[key] = idx
    print('\n=== Derm Foundation Embeddings ===')
    print(f'Embeddings shape: {embeddings.shape}')
    print(f'Total images: {len(embeddings):,}')
    print(f'Total cases: {len(set(case_ids)):,}')
    return (embeddings, embedding_map)

# 전처리 데이터 로딩 및 검증
def load_split(data_dir, split, embeddings, embedding_map, num_classes):
    # Manifest에는 이미지 식별자, NPZ에는 학습에 필요한 배열이 저장됨
    manifest_path = data_dir / f'{split}_manifest.csv'
    arrays_path = data_dir / f'{split}_arrays.npz'

    df = pd.read_csv(manifest_path, dtype={'case_id': str, 'image_col': str})
    required = {'case_id', 'image_col', 'image_path', 'case_weight', 'top10_mass', 'hard_label', 'split'}

    if not required.issubset(df.columns):
        raise ValueError(f'{split} 필수 컬럼 누락: {required - set(df.columns)}')
    if df[list(required)].isna().any().any():
        raise ValueError(f'{split} 필수 컬럼에 결측치가 있습니다.')
    if len(df) == 0:
        raise ValueError(f'{split} 데이터셋이 비어 있습니다.')
    if not df['split'].eq(split).all():
        raise ValueError(f'{split} Manifest의 split 값 오류')
    if not df['image_col'].str.fullmatch('image_[123]_path').all():
        raise ValueError(f'{split} image_col 형식 오류')

    df['image_id'] = df['case_id'] + '_' + df['image_col'].str.replace('_path$', '', regex=True)

    if df.duplicated(['case_id', 'image_id']).any():
        raise ValueError(f'{split} 중복 이미지 존재')

    with np.load(arrays_path) as data:
        required_arrays = {'soft_target', 'case_weight', 'top10_mass'}
        if not required_arrays.issubset(data.files):
            raise ValueError(f'{split} 필수 배열 누락: {required_arrays - set(data.files)}')
        targets = np.asarray(data['soft_target'], dtype=np.float32)
        weights = np.asarray(data['case_weight'], dtype=np.float32).reshape(-1)
        top10_mass = np.asarray(data['top10_mass'], dtype=np.float32).reshape(-1)
        print(f'\n=== {split.upper()} NPZ ===')
        for key in data.files:
            print(f'{key}: shape={data[key].shape}, dtype={data[key].dtype}')

    if not len(df) == len(targets) == len(weights) == len(top10_mass):
        raise ValueError(f'{split} Manifest와 NPZ 개수 불일치')
    if not np.allclose(df['case_weight'].to_numpy(dtype=np.float32), weights, atol=1e-06):
        raise ValueError(f'{split} Manifest와 NPZ case_weight 불일치')
    if not np.allclose(df['top10_mass'].to_numpy(dtype=np.float32), top10_mass, atol=1e-06):
        raise ValueError(f'{split} Manifest와 NPZ top10_mass 불일치')
    if targets.shape != (len(df), num_classes):
        raise ValueError(f'{split} Soft Label 차원 오류: {targets.shape}')
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError(f'{split} Soft Label 값 오류')
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-05):
        raise ValueError(f'{split} Soft Label 확률 합 오류')
    if not np.isfinite(weights).all() or (weights <= 0).any() or (weights > 1).any():
        raise ValueError(f'{split} Case Weight 값 오류')
    if not np.isfinite(top10_mass).all() or (top10_mass < 0).any() or (top10_mass > 1 + 1e-06).any():
        raise ValueError(f'{split} Top10 Mass 값 오류')

    raw_hard_labels = pd.to_numeric(df['hard_label'], errors='raise').to_numpy()
    if not np.isfinite(raw_hard_labels).all() or (raw_hard_labels != np.floor(raw_hard_labels)).any():
        raise ValueError(f'{split} Hard Label 형식 오류')
    hard_labels = raw_hard_labels.astype(np.int64)
    if (hard_labels < -1).any() or (hard_labels >= num_classes).any():
        raise ValueError(f'{split} Hard Label 범위 오류')

    embedding_indices = []
    missing = []
    for cid, iid in zip(df['case_id'], df['image_id']):
        key = (cid, iid)
        if key not in embedding_map:
            missing.append(key)
        else:
            embedding_indices.append(embedding_map[key])
    if missing:
        raise ValueError(f'{split}: 기존 임베딩이 없는 이미지 {len(missing)}개\n예시: {missing[:5]}')

    split_embeddings = embeddings[np.asarray(embedding_indices, dtype=np.int64)]
    first_case_index = {}
    for i, cid in enumerate(df['case_id']):
        if cid not in first_case_index:
            first_case_index[cid] = i
            continue
        first_idx = first_case_index[cid]
        if not np.allclose(targets[first_idx], targets[i]):
            raise ValueError(f'동일 케이스 Soft Label 불일치: {cid}')
        if hard_labels[first_idx] != hard_labels[i]:
            raise ValueError(f'동일 케이스 Hard Label 불일치: {cid}')
        if not np.isclose(top10_mass[first_idx], top10_mass[i]):
            raise ValueError(f'동일 케이스 Top10 Mass 불일치: {cid}')

    weight_sums = df.groupby('case_id')['case_weight'].sum()
    if not np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-05):
        print(f'WARNING: {split} 케이스별 Weight 합이 모두 1은 아닙니다.')

    df['hard_label'] = hard_labels
    case_df = df.drop_duplicates('case_id')
    valid_cases = (case_df['hard_label'] >= 0).sum()
    invalid_cases = (case_df['hard_label'] == -1).sum()
    print(f'\n=== {split.upper()} Dataset ===')
    print(f"Cases: {df['case_id'].nunique():,}")
    print(f'Images: {len(df):,}')
    print(f'Matched embeddings: {len(split_embeddings):,}')
    print(f'Embeddings shape: {split_embeddings.shape}')
    print(f'Targets shape: {targets.shape}')
    print(f'Weights shape: {weights.shape}')
    print(f'Valid hard-label cases: {valid_cases:,}')
    print(f'Hard-label -1 cases: {invalid_cases:,}')
    print('\nCase Weight Sum:')
    print(weight_sums.describe())
    print('\nHard Label Distribution:')
    print(case_df['hard_label'].value_counts().sort_index())
    dataset = EmbeddingDataset(embeddings=split_embeddings, targets=targets, weights=weights, hard_labels=hard_labels, case_ids=df['case_id'].tolist())
    return (dataset, set(df['case_id']))

# 기존 weighted_soft_ce 기반 학습
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_numerator = 0.0
    total_denominator = 0.0
    for batch in loader:
        image = batch['image'].to(device)
        target = batch['target'].to(device)
        weight = batch['weight'].to(device)

        optimizer.zero_grad()
        logits = model(image)
        loss = weighted_soft_ce(logits, target, weight, class_weights=None)
        if not torch.isfinite(loss):
            raise ValueError('학습 중 Loss가 NaN 또는 Inf입니다.')

        loss.backward()
        optimizer.step()
        weight_sum = weight.sum().item()
        total_numerator += loss.item() * weight_sum
        total_denominator += weight_sum
    return total_numerator / max(total_denominator, 1e-08)

# 케이스 단위 평가: hard_label=-1은 hard 지표에서 제외
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    case_probs = defaultdict(list)
    case_hard_labels = {}
    case_soft_targets = {}
    total_numerator = 0.0
    total_denominator = 0.0
    for batch in loader:
        image = batch['image'].to(device)
        target = batch['target'].to(device)
        weight = batch['weight'].to(device)
        logits = model(image)
        loss = weighted_soft_ce(logits, target, weight, class_weights=None)

        weight_sum = weight.sum().item()
        total_numerator += loss.item() * weight_sum
        total_denominator += weight_sum

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        hard_labels = batch['hard_label'].cpu().numpy()
        soft_targets = batch['target'].cpu().numpy()

        for cid, p, h, s in zip(batch['case_id'], probs, hard_labels, soft_targets):
            case_probs[cid].append(p)
            case_hard_labels[cid] = int(h)
            case_soft_targets[cid] = s

    all_case_ids = list(case_probs.keys())
    all_probs = np.stack([np.mean(case_probs[cid], axis=0) for cid in all_case_ids])
    all_hard_labels = np.array([case_hard_labels[cid] for cid in all_case_ids])
    all_soft_targets = np.stack([case_soft_targets[cid] for cid in all_case_ids])

    valid_mask = all_hard_labels >= 0
    if not valid_mask.any():
        raise ValueError('평가 가능한 Hard Label 케이스가 없습니다.')
    y_true = all_hard_labels[valid_mask]
    y_probs = all_probs[valid_mask]
    
    y_pred = y_probs.argmax(axis=1)
    labels = list(range(num_classes))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    ranked_predictions = np.argsort(-y_probs, axis=1, kind='stable')
    top3 = ranked_predictions[:, :3]
    top5 = ranked_predictions[:, :5]
    top3_accuracy = np.mean(np.any(top3 == y_true[:, None], axis=1))
    top5_accuracy = np.mean(np.any(top5 == y_true[:, None], axis=1))
    all_ranked = np.argsort(-all_probs, axis=1, kind='stable')
    soft_top3_mass = np.take_along_axis(all_soft_targets, all_ranked[:, :3], axis=1).sum(axis=1).mean()
    soft_top5_mass = np.take_along_axis(all_soft_targets, all_ranked[:, :5], axis=1).sum(axis=1).mean()
    
    precision, recall, class_f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {'loss': total_numerator / max(total_denominator, 1e-08), 'accuracy': float(accuracy), 'macro_f1': float(macro_f1), 'top3_accuracy': float(top3_accuracy), 'top5_accuracy': float(top5_accuracy), 'soft_top3_mass': float(soft_top3_mass), 'soft_top5_mass': float(soft_top5_mass), 'class_precision': precision, 'class_recall': recall, 'class_f1': class_f1, 'class_support': support, 'total_cases': int(len(all_case_ids)), 'evaluated_cases': int(valid_mask.sum()), 'excluded_cases': int((~valid_mask).sum())}

# 클래스별 평가 결과
def print_class_results(metrics, class_map):
    idx_to_class = {idx: name for name, idx in class_map.items()}
    class_results = pd.DataFrame({'class_name': [idx_to_class[i] for i in range(len(class_map))], 'precision': metrics['class_precision'], 'recall': metrics['class_recall'], 'f1': metrics['class_f1'], 'support': metrics['class_support']})
    print('\n=== 클래스별 성능 ===')
    print(class_results.sort_values('f1').to_string(index=False))
    return class_results

def print_metrics(metrics):
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"Soft Top-3 Mass: {metrics['soft_top3_mass']:.4f}")
    print(f"Soft Top-5 Mass: {metrics['soft_top5_mass']:.4f}")
    print(f"Evaluated cases: {metrics['evaluated_cases']}/{metrics['total_cases']}")
    print(f"Excluded cases: {metrics['excluded_cases']}")
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--class-map', type=Path, default=Path('config/class_to_idx_top10.json'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=Path('outputs/derm_top10'))
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.patience) <= 0:
        raise ValueError('학습 설정 값 오류')

    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # 클래스 매핑
    class_map = load_class_to_idx(
        args.class_map,
        require_other=False
    )

    if len(class_map) != 10:
        raise ValueError('클래스 개수가 10개여야 합니다.')
    if sorted(class_map.values()) != list(range(10)):
        raise ValueError('클래스 인덱스 오류')
    if 'other' in class_map:
        raise ValueError('other 클래스가 포함되어 있습니다.')

    num_classes = len(class_map)

    # 전처리 설정 및 클래스 순서 검증
    config_path = args.data_dir / 'preprocessing_config.json'
    preprocessing_config = None

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            preprocessing_config = json.load(f)

        if 'top10_classes' in preprocessing_config:
            ordered_classes = sorted(class_map, key=class_map.get)

            if ordered_classes != preprocessing_config['top10_classes']:
                raise ValueError('클래스 순서가 전처리 설정과 다릅니다.')

    # 임베딩 로딩
    embeddings, embedding_map = load_embeddings(args.embeddings)
    embedding_dim = embeddings.shape[1]

    # 데이터셋 구성
    datasets = {}
    split_cases = {}

    for split in ['train', 'validation']:
        dataset, cases = load_split(
            data_dir=args.data_dir,
            split=split,
            embeddings=embeddings,
            embedding_map=embedding_map,
            num_classes=num_classes
        )

        datasets[split] = dataset
        split_cases[split] = cases

    # Test 데이터는 학습에 사용하지 않고 case_id만 읽어 데이터 누수 검증
    split_cases['test'] = set(pd.read_csv(
        args.data_dir / 'test_manifest.csv',
        usecols=['case_id'],
        dtype={'case_id': str}
    )['case_id'])

    # 데이터 누수 검증
    for split_a, split_b in [
        ('train', 'validation'),
        ('train', 'test'),
        ('validation', 'test')
    ]:
        overlap = split_cases[split_a] & split_cases[split_b]

        if overlap:
            raise ValueError(
                f'{split_a}와 {split_b} 간 case_id 중복: {len(overlap)}개'
            )

    # DataLoader 구성
    train_loader = DataLoader(
        datasets['train'],
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        datasets['validation'],
        batch_size=args.batch_size,
        shuffle=False
    )

    # 모델 및 Optimizer 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DermClassifier(
        input_dim=embedding_dim,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f'Device: {device}')

    # 학습 및 Early Stopping
    checkpoint = args.out / 'derm_top10_best.pt'
    best = float('-inf')
    best_epoch = 0
    best_metrics = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device
        )
        metrics = evaluate(
            model, val_loader, device, num_classes
        )

        val_f1 = metrics['macro_f1']

        if not np.isfinite(val_f1):
            raise ValueError('Validation Macro F1 값 오류')

        # Epoch별 성능 출력
        print(f"\n{'=' * 50}")
        print(f'Epoch {epoch}/{args.epochs}')
        print('=' * 50)

        print(f'Train Loss: {train_loss:.4f}')
        print_metrics(metrics)
        print_class_results(metrics, class_map)

        # 최고 성능 모델 저장
        if val_f1 > best:
            best = val_f1
            best_epoch = epoch
            best_metrics = metrics
            stale = 0

            torch.save({
                'model': model.state_dict(),
                'class_map': class_map,
                'embedding_dim': embedding_dim,
                'num_classes': num_classes,
                'epoch': epoch,
                'val_macro_f1': best,
                'val_accuracy': metrics['accuracy'],
                'val_top3_accuracy': metrics['top3_accuracy'],
                'val_top5_accuracy': metrics['top5_accuracy'],
                'val_soft_top3_mass': metrics['soft_top3_mass'],
                'val_soft_top5_mass': metrics['soft_top5_mass'],
                'preprocessing_config': preprocessing_config,
                'config': {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                }
            }, checkpoint)

            print(f'\nBest model saved! Macro F1: {best:.4f}')

        else:
            stale += 1

            print(f'\nEarly stopping counter: {stale}/{args.patience}')

            if stale >= args.patience:
                print('Early stopping')
                break

    # 최종 학습 결과
    print('\n=== Training Completed ===')
    print(f'Best Epoch: {best_epoch}')
    print_metrics(best_metrics)
    print_class_results(best_metrics, class_map)
    print(f'\nCheckpoint: {checkpoint}')



if __name__ == '__main__':
    main()