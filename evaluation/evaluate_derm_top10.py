import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.backbones.derm_foundation import DermClassifier
from training.train_derm_top10 import (
    load_embeddings,
    load_split,
    evaluate,
    print_metrics,
    print_class_results
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    saved = torch.load(args.checkpoint, map_location=device)
    class_map = saved['class_map']
    num_classes = saved['num_classes']
    embedding_dim = saved['embedding_dim']

    embeddings, embedding_map = load_embeddings(args.embeddings)

    if embeddings.shape[1] != embedding_dim:
        raise ValueError('체크포인트와 임베딩 차원이 일치하지 않습니다.')

    # Test 데이터셋의 클래스 순서 검증
    config_path = args.data_dir / 'preprocessing_config.json'

    if config_path.exists():
        import json

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if 'top10_classes' in config:
            ordered_classes = sorted(class_map, key=class_map.get)

            if ordered_classes != config['top10_classes']:
                raise ValueError('학습 모델과 Test 데이터의 클래스 순서가 다릅니다.')

    # Test 데이터셋만 로딩
    test_dataset, _ = load_split(
        data_dir=args.data_dir,
        split='test',
        embeddings=embeddings,
        embedding_map=embedding_map,
        num_classes=num_classes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = DermClassifier(
        input_dim=embedding_dim,
        num_classes=num_classes
    ).to(device)

    # 저장된 모델 가중치 복원
    model.load_state_dict(saved['model'])

    # 기존 평가 함수를 재사용하여 Test 성능 측정
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        num_classes
    )

    print('\n=== Test Results ===')
    print_metrics(test_metrics)
    print_class_results(test_metrics, class_map)


if __name__ == '__main__':
    main()