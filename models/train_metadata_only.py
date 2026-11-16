"""SCIN 34-feature Metadata-only ablation, no images or torchvision required.

Example (Colab, from /content/data):
 python models/train_metadata_only.py \
    --csv '/content/drive/MyDrive/데이터사이언스응용 - 고양이 팀/integration/multimodal_integrated_dataset.csv' \
    --class-map-checkpoint outputs/image_best.pt \
    --output '/content/drive/MyDrive/데이터사이언스응용 - 고양이 팀/outputs/metadata_only_best.pt' \
    --epochs 40 --batch-size 64 --lr 0.0001 --patience 8 --seed 42 --class-weighting

Each case has the same metadata replicated across its images. We deduplicate
by case_id before training and validation to avoid unintended reweighting.
case-level metrics: 33-class Macro F1 (zero_division=0), accuracy.
The held-out test split is not evaluated or used to select models.
"""
from __future__ import annotations
import argparse
import ast
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

FEATURE_COLS = [
    'textures_raised_or_bumpy', 'textures_flat', 'textures_rough_or_flaky', 'textures_fluid_filled',
    'body_parts_head_or_neck', 'body_parts_arm', 'body_parts_palm', 'body_parts_back_of_hand',
    'body_parts_torso_front', 'body_parts_torso_back', 'body_parts_genitalia_or_groin',
    'body_parts_buttocks', 'body_parts_leg', 'body_parts_foot_top_or_side', 'body_parts_foot_sole',
    'body_parts_other', 'condition_symptoms_bleeding', 'condition_symptoms_increasing_size',
    'condition_symptoms_darkening', 'condition_symptoms_itching', 'condition_symptoms_burning',
    'condition_symptoms_pain', 'textures_has_recorded_yes', 'body_parts_has_recorded_yes',
    'condition_symptoms_has_recorded_yes', 'duration_LESS_THAN_ONE_WEEK',
    'duration_MORE_THAN_FIVE_YEARS', 'duration_MORE_THAN_ONE_YEAR',
    'duration_ONE_DAY', 'duration_ONE_TO_FOUR_WEEKS', 'duration_ONE_TO_THREE_MONTHS',
    'duration_SINCE_CHILDHOOD', 'duration_THREE_TO_TWELVE_MONTHS', 'duration_UNKNOWN'
]
assert len(FEATURE_COLS) == 34
NUM_CLASSES = 33


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_soft(raw):
    target = ast.literal_eval(raw) if isinstance(raw, str) else raw
    arr = np.asarray(target, dtype=np.float32)
    if arr.shape != (NUM_CLASSES,) or not np.isfinite(arr).all() or (arr < 0).any() or arr.sum() <= 0:
        raise ValueError(f'Invalid soft_label: expected positive finite {NUM_CLASSES}-vector, got {arr.shape}')
    return arr / arr.sum()


def load_and_check_csv(path):
    frame = pd.read_csv(path, dtype={'case_id': str})
    required = {'case_id', 'split', 'soft_label', 'sample_weight', *FEATURE_COLS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    if frame[['case_id', 'split']].isna().any().any():
        raise ValueError('Missing case_id / split')
    if not set(frame.split.unique()).issubset({'train', 'validation', 'test'}):
        raise ValueError(f'Unexpected splits: {frame.split.unique().tolist()}')
    if frame.groupby('case_id').split.nunique().gt(1).any():
        raise ValueError('Data leakage: same case_id appears in multiple splits')
    # All observations of a case must have identical targets and metadata.
    # NaN grouping uses dropna=False and is handled downstream via masks.
    for col in ['soft_label', 'sample_weight', *FEATURE_COLS]:
        inconsistent = frame.groupby('case_id')[col].nunique(dropna=False).gt(1)
        if inconsistent.any():
            raise ValueError(f'Conflicting {col} within case_id {inconsistent[inconsistent].index[0]}')
    images = frame.split.value_counts()
    print('Split images:', {k: int(images.get(k, 0)) for k in ('train', 'validation', 'test')}, flush=True)
    # Never let duplicate images alter metadata-only sample weighting.
    cases = frame.drop_duplicates('case_id').copy().reset_index(drop=True)
    print('Split cases:', {k: int((cases.split == k).sum()) for k in ('train', 'validation', 'test')}, flush=True)
    if not {'train', 'validation'}.issubset(set(cases.split.unique())):
        raise ValueError('Training and validation splits are required')
    return cases


class CaseDataset(Dataset):
    def __init__(self, frame):
        self.df = frame.reset_index(drop=True)
        self.features = self.df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
        if np.isinf(self.features).any():
            raise ValueError('Feature has infinity')
        self.targets = np.stack([parse_soft(s) for s in self.df.soft_label])
        self.weights = pd.to_numeric(self.df.sample_weight, errors='raise').to_numpy(dtype=np.float32)
        if not np.isfinite(self.weights).all() or (self.weights <= 0).any() or (self.weights > 1).any():
            raise ValueError('Invalid sample_weight: expected (0,1]')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]), torch.from_numpy(self.targets[idx]),
                torch.tensor(self.weights[idx]), str(self.df.at[idx, 'case_id']))


class MetadataClassifier(nn.Module):
    """Input masks preserve missing information; logits are 33-way softmax CE targets."""
    def __init__(self, in_dim=34, n_classes=33, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        observed = torch.isfinite(x)
        clean = torch.where(observed, x, torch.zeros_like(x))
        return self.net(torch.cat((clean, observed.to(x.dtype)), dim=1))


def class_weights_from_train(dataset):
    class_mass = (dataset.targets * dataset.weights[:, None]).sum(axis=0)
    weights = 1. / np.sqrt(class_mass + 1e-8)
    weights /= weights.mean()
    return torch.as_tensor(np.clip(weights, 0.5, 3.0), dtype=torch.float32)


def weighted_soft_ce(logits, targets, sample_weights, class_weights=None):
    logp = F.log_softmax(logits, dim=1)
    weighted_target = targets if class_weights is None else targets * class_weights.unsqueeze(0)
    loss_per_case = -(weighted_target * logp).sum(dim=1)
    return (loss_per_case * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def run_epoch(model, loader, device, optimizer=None, class_weights=None, collect=False):
    training = optimizer is not None
    model.train(training)
    total_loss_weighted, total_weights = 0.0, 0.0
    rows = []
    for features, targets, weights, case_ids in loader:
        features = features.to(device)
        targets = targets.to(device)
        weights = weights.to(device)
        with torch.set_grad_enabled(training):
            logits = model(features)
            # The team's baseline validates without training class weights.
            loss = weighted_soft_ce(logits, targets, weights, class_weights if training else None)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        wsum = float(weights.sum().item())
        total_loss_weighted += loss.item() * wsum
        total_weights += wsum
        if collect:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            true = targets.argmax(dim=1).detach().cpu().tolist()
            rows.extend((cid, p, t) for cid, p, t in zip(case_ids, probs, true))
    return total_loss_weighted / max(total_weights, 1e-8), rows


def evaluate(records):
    # Each case occurs once. Explicit grouping retained as a safety guard.
    data = {}
    for cid, prob, target in records:
        if cid in data:
            raise ValueError(f'Metadata-only evaluation repeated case_id: {cid}')
        data[cid] = (prob, target)
    if not data:
        raise ValueError('No validation records')
    y_true = np.array([t for _, t in data.values()], dtype=int)
    y_pred = np.array([int(p.argmax()) for p, _ in data.values()], dtype=int)
    metrics = dict(cases=len(y_true),
                   macro_f1=float(f1_score(y_true, y_pred, average='macro', labels=list(range(NUM_CLASSES)), zero_division=0)),
                   accuracy=float(accuracy_score(y_true, y_pred)))
    return metrics, y_true, y_pred


def save_analysis(y_true, y_pred, train_dataset, class_map, output):
    labels = list(range(NUM_CLASSES))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    inverse = {int(v): k for k, v in class_map.items()}
    train_hard = train_dataset.targets.argmax(axis=1)
    counts = np.bincount(train_hard, minlength=NUM_CLASSES)
    table = pd.DataFrame({
        'class_idx': labels, 'class_name': [inverse.get(i, f'class_{i}') for i in labels],
        'train_cases': counts, 'val_support': support, 'precision': prec, 'recall': rec, 'f1': f1,
    })
    path = output.with_name(output.stem + '_class_f1.csv')
    table.to_csv(path, index=False, encoding='utf-8-sig')
    print('Class F1 CSV:', path, flush=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_rows = [(inverse.get(i, str(i)), inverse.get(j, str(j)), int(cm[i, j]))
                      for i in labels for j in labels if i != j and cm[i, j] > 0]
    confusion = pd.DataFrame(confusion_rows, columns=['actual', 'predicted', 'count'])
    confusion = confusion.sort_values('count', ascending=False)
    confuse_path = output.with_name(output.stem + '_confusions.csv')
    confusion.to_csv(confuse_path, index=False, encoding='utf-8-sig')
    print('Confusions CSV:', confuse_path, flush=True)
    print('\nTop classes by F1:')
    print(table.sort_values('f1', ascending=False).head(10).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\nCounts: absent validation labels =', int((table.val_support == 0).sum()),
          '; nonzero support but F1=0 =', int(((table.val_support > 0) & (table.f1 == 0)).sum()))


def main():
    parser = argparse.ArgumentParser(description='Standalone SCIN metadata-only MLP (34 input features, 33 classes)')
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--class-map-checkpoint', type=Path, default=None,
                        help='Image-only checkpoint, read ONLY for original 33-class name/index mapping')
    parser.add_argument('--output', type=Path, default=Path('outputs/metadata_only_best.pt'))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--class-weighting', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError('epochs, batch-size, patience must be positive')
    set_seed(args.seed)
    cases = load_and_check_csv(args.csv)
    split_frames = {s: cases.loc[cases.split == s].copy() for s in ('train', 'validation')}
    datasets = {s: CaseDataset(split_frames[s]) for s in split_frames}
    class_map = {f'class_{i}': i for i in range(NUM_CLASSES)}
    if args.class_map_checkpoint is not None:
        checkpoint = torch.load(args.class_map_checkpoint, map_location='cpu', weights_only=True)
        class_map = checkpoint['class_map']
        if sorted(map(int, class_map.values())) != list(range(NUM_CLASSES)):
            raise ValueError('Class map does not contain exactly indices 0..32')
        print('Class names loaded from checkpoint:', args.class_map_checkpoint, flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    print('Input features:', len(FEATURE_COLS), '(metadata only; no images loaded)', flush=True)
    loaders = {s: DataLoader(datasets[s], batch_size=args.batch_size, shuffle=(s == 'train'),
                             num_workers=args.workers, pin_memory=(device.type == 'cuda')) for s in datasets}
    model = MetadataClassifier(len(FEATURE_COLS), NUM_CLASSES, args.dropout).to(device)
    class_weights = class_weights_from_train(datasets['train']).to(device) if args.class_weighting else None
    print('Class weighting:', 'enabled' if class_weights is not None else 'disabled', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, loaders['train'], device, optimizer, class_weights)
        scheduler.step()
        val_loss, val_records = run_epoch(model, loaders['validation'], device, collect=True)
        metrics, y_true, y_pred = evaluate(val_records)
        print(f'Epoch {epoch:02d}/{args.epochs} | lr={optimizer.param_groups[0]["lr"]:.7f} '
              f'| train_loss={train_loss:.4f} | val_loss={val_loss:.4f} '
              f'| val_case_f1={metrics["macro_f1"]:.4f} | val_case_acc={metrics["accuracy"]:.4f} '
              f'| cases={metrics["cases"]}', flush=True)
        if metrics['macro_f1'] > best_f1:
            best_f1, best_epoch, stale = metrics['macro_f1'], epoch, 0
            torch.save({
                'model_state_dict': model.state_dict(), 'epoch': epoch,
                'val_macro_f1': best_f1, 'val_loss': val_loss,
                'val_case_accuracy': metrics['accuracy'], 'val_cases': metrics['cases'],
                'class_map': class_map, 'symptom_cols': FEATURE_COLS,
                'num_classes': NUM_CLASSES, 'evaluation_level': 'case', 'mode': 'metadata_only',
                'seed': args.seed, 'config': {**vars(args), 'csv': str(args.csv),
                                              'output': str(args.output),
                                              'class_map_checkpoint': str(args.class_map_checkpoint)},
            }, args.output)
            save_analysis(y_true, y_pred, datasets['train'], class_map, args.output)
            print('  Saved best:', args.output, flush=True)
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping', flush=True)
                break
    print(f'BEST epoch={best_epoch} | val_case_macro_f1={best_f1:.4f}', flush=True)
    print('Saved checkpoint:', args.output, flush=True)
    print('NOTE: Metadata-only is trained per case (not replicated image rows). '
          'Same case split/33-label metric; training sampling differs intentionally.', flush=True)


if __name__ == '__main__':
    main()
