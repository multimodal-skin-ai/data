
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from models import ImageClassifier, MultimodalClassifier
from training.train import SCINDataset, run_epoch, case_metrics


# ========================================
# 저장된 최적 모델 불러오기 -> Test 데이터 로딩
# -> 모델 예측 -> 케이스 단위 최종 성능 평가
# ========================================
def main():
    # 1. 실행 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', type=Path, default=Path('datasets/processed/test.csv'))
    parser.add_argument('--image-root', type=Path, default=Path('.'))
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/image_best.pt'))
    parser.add_argument('--batch-size', type=int, default=16)

    args = parser.parse_args()

    # 2. 학습된 모델 정보 불러오기
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = torch.load(
        args.checkpoint,
        map_location='cpu',
        weights_only=True
    )

    mapping = state['class_map']
    symptom_cols = state['symptom_cols']
    mode = state['mode']

    # 3. Test 데이터 불러오기 및 검증
    test_df = pd.read_csv(
        args.test_csv,
        dtype={'case_id': str}
    )

    if 'split' in test_df.columns:
        test_df = test_df[
            test_df['split'] == 'test'
        ].copy()

    if test_df.empty:
        raise ValueError('Test 데이터가 없습니다.')

    required = {
        'case_id',
        'image_path',
        'soft_label',
        'sample_weight',
        *symptom_cols
    }

    missing = required - set(test_df.columns)

    if missing:
        raise ValueError(
            f'Test CSV missing columns: {missing}'
        )

    if test_df[list(required)].isna().any().any():
        raise ValueError(
            'Test CSV contains missing required values'
        )

    # 4. Test Dataset 및 DataLoader 생성
    test_dataset = SCINDataset(
        test_df,
        args.image_root,
        mapping,
        symptom_cols,
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 5. 모델 생성 및 학습된 가중치 적용
    if mode == 'image':
        model = ImageClassifier(
            num_classes=len(mapping),
            pretrained=False
        )

    elif mode == 'multimodal':
        model = MultimodalClassifier(
            num_classes=len(mapping),
            symptom_dim=len(symptom_cols),
            pretrained=False,
            freeze_backbone=True
        )

    else:
        raise ValueError(f'Unknown model mode: {mode}')

    model.load_state_dict(state['model'])
    model = model.to(device)

    # 6. Test 데이터로 최종 성능 평가
    test_loss, test_records = run_epoch(
        model,
        test_loader,
        device
    )

    metrics = case_metrics(test_records)

    print('\n===== TEST RESULTS =====')
    print(f'Model: {mode}')
    print(f'Test cases: {metrics["cases"]}')
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {metrics["accuracy"]:.4f}')
    print(f'Test Macro F1: {metrics["macro_f1"]:.4f}')

    # 7. 클래스별 F1 계산
    grouped = {}

    for cid, prob, label in test_records:
        if cid not in grouped:
            grouped[cid] = {
                'probs': [],
                'label': label
            }

        if grouped[cid]['label'] != label:
            raise ValueError(
                f'Inconsistent targets for case {cid}'
            )

        grouped[cid]['probs'].append(prob)

    y_true = []
    y_pred = []

    for item in grouped.values():
        y_true.append(item['label'])

        mean_prob = np.mean(
            item['probs'],
            axis=0
        )

        y_pred.append(
            int(mean_prob.argmax())
        )

    labels = list(range(len(mapping)))

    class_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )

    idx_to_class = {
        idx: name
        for name, idx in mapping.items()
    }

    print('\n===== CLASS-WISE F1 =====')

    for idx, score in zip(labels, class_f1):
        print(
            f'{idx_to_class[idx]}: {score:.4f}'
        )


if __name__ == '__main__':
    main()
