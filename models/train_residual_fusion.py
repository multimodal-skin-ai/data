"""Standalone SCIN residual-fusion trainer (EfficientNet-B3 image-only checkpoint).

Requires only torch, torchvision, numpy, pandas, pillow, scikit-learn.
No dependency on models/backbone.py, multimodal.py, train_multimodal.py.

Example:
python train_residual_fusion.py --csv "/content/drive/MyDrive/데이터사이언스응용 - 고양이 팀/integration/multimodal_integrated_dataset.csv" --image-root /content/images --checkpoint /content/data/outputs/image_best.pt --output /content/data/outputs/residual_fusion_best.pt
"""
from __future__ import annotations
import argparse
import ast
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# Same 34 columns as the provided symptoms_all.csv; deliberately do not treat
# administrative/label columns in final_integrated_dataset.csv as symptoms.
FEATURE_COLS = [
    'textures_raised_or_bumpy','textures_flat','textures_rough_or_flaky','textures_fluid_filled',
    'body_parts_head_or_neck','body_parts_arm','body_parts_palm','body_parts_back_of_hand',
    'body_parts_torso_front','body_parts_torso_back','body_parts_genitalia_or_groin',
    'body_parts_buttocks','body_parts_leg','body_parts_foot_top_or_side','body_parts_foot_sole','body_parts_other',
    'condition_symptoms_bleeding','condition_symptoms_increasing_size','condition_symptoms_darkening',
    'condition_symptoms_itching','condition_symptoms_burning','condition_symptoms_pain',
    'textures_has_recorded_yes','body_parts_has_recorded_yes','condition_symptoms_has_recorded_yes',
    'duration_LESS_THAN_ONE_WEEK','duration_MORE_THAN_FIVE_YEARS','duration_MORE_THAN_ONE_YEAR',
    'duration_ONE_DAY','duration_ONE_TO_FOUR_WEEKS','duration_ONE_TO_THREE_MONTHS',
    'duration_SINCE_CHILDHOOD','duration_THREE_TO_TWELVE_MONTHS','duration_UNKNOWN'
]
assert len(FEATURE_COLS) == 34


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_soft(raw, n):
    obj = ast.literal_eval(raw) if isinstance(raw, str) else raw
    arr = np.asarray(obj, dtype=np.float32)
    if arr.shape != (n,) or not np.isfinite(arr).all() or np.any(arr < 0) or arr.sum() <= 0:
        raise ValueError(f'Invalid soft_label with shape={arr.shape}, expected {n}')
    return arr / arr.sum()


def resolve_image_paths(frame: pd.DataFrame, image_root: Path | None) -> pd.DataFrame:
    """Use original image_path if it exists; otherwise resolve from image_root/split/image_id.

    Intentionally do not fall back to lowres/blur/dark test images.
    """
    frame = frame.copy()
    resolved = []
    for row in frame.itertuples(index=False):
        original = str(getattr(row, 'image_path', ''))
        if original and original != 'nan' and Path(original).is_file():
            resolved.append(original)
            continue
        if image_root is None:
            raise FileNotFoundError(f'Image path unavailable: {original}; pass --image-root')
        split = str(row.split)
        name = str(row.image_id)
        names = [name] if Path(name).suffix else [name + '.jpg', name + '.jpeg', name + '.png']
        subdirs = {
            'train': ['train_images/original', 'train/train_images/original', 'train_images'],
            'validation': ['validation_images', 'validation/validation_images', 'validation_images/original'],
            'test': ['test_images/original', 'test_images', 'test/test_images/original'],
        }.get(split, [])
        candidates = [image_root / sub / fn for sub in subdirs for fn in names]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Image not found for split={split}, image_id={name}. ' 
                                    f'Checked: {[str(p) for p in candidates[:4]]}; inspect --image-root')
        resolved.append(str(path))
    frame['image_path'] = resolved
    return frame


class SCINDataset(Dataset):
    def __init__(self, frame, n_classes, transform):
        self.df = frame.reset_index(drop=True)
        self.transform = transform
        self.symptoms = self.df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
        self.soft = np.stack([parse_soft(x, n_classes) for x in self.df.soft_label])
        self.weights = pd.to_numeric(self.df.sample_weight, errors='raise').to_numpy(dtype=np.float32)
        if np.any(~np.isfinite(self.weights)) or np.any(self.weights <= 0) or np.any(self.weights > 1):
            raise ValueError('Invalid sample_weight: expected finite values in (0,1]')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        with Image.open(self.df.at[i, 'image_path']) as img:
            image = self.transform(img.convert('RGB'))
        return (image, torch.from_numpy(self.symptoms[i]),
                torch.from_numpy(self.soft[i]), torch.tensor(self.weights[i], dtype=torch.float32),
                str(self.df.at[i, 'case_id']))


class SymptomEncoder(nn.Module):
    def __init__(self, in_dim=34, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, out_dim), nn.ReLU())

    def forward(self, x):
        observed = torch.isfinite(x)
        present = observed.any(dim=1, keepdim=True)
        clean = torch.where(observed, x, torch.zeros_like(x))
        inputs = torch.cat([clean, observed.float(), present.float()], dim=1)
        return self.net(inputs) * present.float()


class ResidualFusion(nn.Module):
    """Preserve pretrained image logits; learn additive symptom-conditioned correction.

    At initialization correction == 0 -> output is exactly the image-only model.
    """
    def __init__(self, n_classes, feature_dim=1536, symptom_dim=34):
        super().__init__()
        network = models.efficientnet_b3(weights=None)
        actual_dim = network.classifier[1].in_features
        if actual_dim != feature_dim:
            raise RuntimeError(f'Expected image feature_dim={feature_dim}; got={actual_dim}')
        network.classifier = nn.Identity()
        self.backbone = network
        self.image_head = nn.Linear(feature_dim, n_classes)
        self.symptom_encoder = SymptomEncoder(symptom_dim, 64)
        self.correction = nn.Sequential(
            nn.Linear(feature_dim + 64, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
        # Critical: residual model starts at exact image-only predictions.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def freeze_image(self):
        self.backbone.requires_grad_(False)
        self.image_head.requires_grad_(False)
        self.backbone.eval()
        self.image_head.eval()

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()  # do not drift pretrained BN running stats
        if not any(p.requires_grad for p in self.image_head.parameters()):
            self.image_head.eval()
        return self

    def forward(self, images, symptoms):
        # Frozen network: avoid storing backbone graph in GPU memory.
        if not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                visual = self.backbone(images)
                image_logits = self.image_head(visual)
        else:
            visual = self.backbone(images)
            image_logits = self.image_head(visual)
        symptom_features = self.symptom_encoder(symptoms)
        delta = self.correction(torch.cat((visual, symptom_features), dim=1))
        return image_logits + delta


def restore_image_only(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'model' not in ckpt or 'class_map' not in ckpt:
        raise KeyError("Expected image-only checkpoint keys: 'model' and 'class_map'")
    state = ckpt['model']
    backbone_state = {k[len('backbone.model.'):]: v for k,v in state.items() if k.startswith('backbone.model.')}
    if not backbone_state:
        raise ValueError('Checkpoint has no backbone.* weights')
    model.backbone.load_state_dict(backbone_state, strict=True)
    model.image_head.load_state_dict({'weight': state['head.1.weight'], 'bias': state['head.1.bias']}, strict=True)
    return ckpt['class_map']


def compute_class_weights(frame, n_classes):
    targets = np.stack([parse_soft(raw,n_classes) for raw in frame.soft_label])
    confidences = frame.sample_weight.to_numpy(np.float32)
    mass = (targets * confidences[:,None]).sum(axis=0)
    cw = 1 / np.sqrt(mass + 1e-8)
    cw = np.clip(cw/cw.mean(), 0.5, 3.0)
    return torch.from_numpy(cw.astype(np.float32))


def weighted_soft_ce(logits, targets, sample_weights, class_weights=None):
    log_probs = F.log_softmax(logits, dim=1)
    contributions = targets * log_probs
    if class_weights is not None:
        contributions = contributions * class_weights.unsqueeze(0)
    loss_per = -contributions.sum(dim=1)
    return (loss_per * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def epoch_pass(model, loader, device, optimizer=None, class_weights=None, collect=False):
    training = optimizer is not None
    model.train(training)
    records = []
    loss_sum = 0.0
    weight_sum = 0.0
    for images, symptoms, targets, weights, ids in loader:
        images = images.to(device, non_blocking=True)
        symptoms = symptoms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(images, symptoms)
            loss = weighted_soft_ce(logits, targets, weights, class_weights if training else None)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_weight = weights.sum().item()
        loss_sum += loss.item() * batch_weight
        weight_sum += batch_weight
        if collect:
            probabilities = logits.detach().softmax(dim=1).cpu().numpy()
            hard_targets = targets.detach().argmax(dim=1).cpu().numpy()
            records.extend((str(cid), prob, int(target)) for cid, prob, target in zip(ids, probabilities, hard_targets))
    return loss_sum / max(weight_sum, 1e-8), records


def case_evaluate(records, n_classes):
    cases = defaultdict(lambda: {'probabilities': [], 'target': None})
    for cid, prob, target in records:
        item = cases[cid]
        if item['target'] is not None and item['target'] != target:
            raise ValueError(f'Inconsistent labels for case {cid}')
        item['target'] = target
        item['probabilities'].append(prob)
    y_true = np.array([item['target'] for item in cases.values()], dtype=int)
    y_pred = np.array([np.mean(item['probabilities'], axis=0).argmax() for item in cases.values()], dtype=int)
    return {
        'cases': len(cases),
        'macro_f1': float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'y_true': y_true, 'y_pred': y_pred,
    }


def make_loaders(frame, size, batch_size, workers, device, test=False):
    train_transform = transforms.Compose([
        transforms.Resize((size,size)), transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize(MEAN,STD)
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)
    ])
    splits = ['train','validation'] + (['test'] if test else [])
    loaders = {}
    for split in splits:
        subset = frame[frame.split == split].copy()
        if len(subset) == 0:
            raise ValueError(f'Empty split: {split}')
        ds = SCINDataset(subset, 33, train_transform if split == 'train' else eval_transform)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'),
                                    num_workers=workers, pin_memory=device.type=='cuda')
    return loaders


def save_class_report(path, metrics, class_map):
    y_true, y_pred = metrics['y_true'], metrics['y_pred']
    n = len(class_map)
    p,r,f,s = precision_recall_fscore_support(y_true,y_pred,labels=list(range(n)),zero_division=0)
    names = {int(v):k for k,v in class_map.items()}
    pd.DataFrame({'class_index':range(n), 'class_name':[names[i] for i in range(n)],
                  'precision':p,'recall':r,'f1':f,'support':s}).to_csv(path,index=False,encoding='utf-8-sig')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',type=Path,required=True)
    ap.add_argument('--image-root',type=Path,default=Path('/content/images'))
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('/content/data/outputs/residual_fusion_best.pt'))
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--patience',type=int,default=5)
    ap.add_argument('--size',type=int,default=300)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--class-weighting',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--evaluate-test',action='store_true',help='Only after final architecture selection')
    args = ap.parse_args()
    seed_all(args.seed)
    frame = pd.read_csv(args.csv, dtype={'case_id':str, 'image_id':str})
    mandatory = {'case_id','split','image_id','soft_label','sample_weight'} | set(FEATURE_COLS)
    missing = mandatory.difference(frame.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    if frame.groupby('case_id').split.nunique().gt(1).any():
        raise ValueError('Data leakage detected: same case_id across splits')
    if not set(frame.split.unique()).issubset({'train','validation','test'}):
        raise ValueError(f'Invalid split names: {frame.split.unique()}')
    # A case may have several images, but each shares a single target and confidence.
    for cid, grp in frame.groupby('case_id', sort=False):
        if grp.soft_label.nunique() != 1 or grp.sample_weight.nunique() != 1:
            raise ValueError(f'Inconsistent target/weight for case_id={cid}')
    frame = resolve_image_paths(frame, args.image_root)
    print('Split images:',frame.groupby('split').size().to_dict(),flush=True)
    print('Split cases:',frame.groupby('split').case_id.nunique().to_dict(),flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device,flush=True)
    original_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    class_map = original_ckpt['class_map']
    if len(class_map) != 33 or sorted(int(v) for v in class_map.values()) != list(range(33)):
        raise ValueError('Expected image checkpoint with 33 contiguous class IDs')
    model = ResidualFusion(n_classes=len(class_map),symptom_dim=len(FEATURE_COLS))
    restore_image_only(model, args.checkpoint)
    model.freeze_image()
    model.to(device)
    loaders = make_loaders(frame,args.size,args.batch_size,args.workers,device,args.evaluate_test)
    cw = compute_class_weights(frame[frame.split=='train'], 33).to(device) if args.class_weighting else None
    print('Class weighting:', 'enabled' if cw is not None else 'disabled',flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    args.output.parent.mkdir(parents=True,exist_ok=True)

    best_f1 = float('-inf')
    best_epoch = -1
    stale = 0

    def save_best(epoch, metrics):
        torch.save({
            'model_state_dict':model.state_dict(), 'epoch':epoch,
            'val_case_macro_f1':metrics['macro_f1'], 'val_case_accuracy':metrics['accuracy'],
            'class_map':class_map, 'symptom_cols':FEATURE_COLS, 'image_size':args.size,
            'image_checkpoint':str(args.checkpoint), 'architecture':'residual_fusion',
            'image_frozen':True,'seed':args.seed, 'class_weighting':bool(args.class_weighting),
            'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        },args.output)
        save_class_report(args.output.with_name(args.output.stem+'_class_f1.csv'),metrics,class_map)

    # Epoch 0 guarantees an identical initial image-only baseline (no random degradation).
    initial_loss, initial_records = epoch_pass(model,loaders['validation'],device,collect=True)
    initial = case_evaluate(initial_records,33)
    best_f1 = initial['macro_f1']
    best_epoch = 0
    save_best(0,initial)
    print(f"Epoch 00 (image-only initialization) | val_loss={initial_loss:.4f} | "
          f"val_case_f1={initial['macro_f1']:.4f} | acc={initial['accuracy']:.4f}", flush=True)

    for epoch in range(1,args.epochs+1):
        train_loss,_ = epoch_pass(model,loaders['train'],device,optimizer=optimizer,class_weights=cw)
        scheduler.step()
        val_loss, records = epoch_pass(model,loaders['validation'],device,collect=True)
        m = case_evaluate(records,33)
        print(f"Epoch {epoch:02d}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.7f} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_case_f1={m['macro_f1']:.4f} | val_case_acc={m['accuracy']:.4f}",flush=True)
        if m['macro_f1'] > best_f1 + 1e-12:
            best_f1=m['macro_f1']
            best_epoch=epoch
            stale=0
            save_best(epoch,m)
            print('  Saved best:', args.output,flush=True)
        else:
            stale+=1
            if stale>=args.patience:
                print('Early stopping',flush=True)
                break
    print(f'BEST epoch={best_epoch} | val_case_macro_f1={best_f1:.4f}',flush=True)
    print('Best checkpoint:',args.output,flush=True)
    print('Class F1 CSV:',args.output.with_name(args.output.stem+'_class_f1.csv'),flush=True)
    if args.evaluate_test:
        saved = torch.load(args.output,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model_state_dict'],strict=True)
        test_loss,records=epoch_pass(model,loaders['test'],device,collect=True)
        test_m=case_evaluate(records,33)
        print(f"FINAL TEST: loss={test_loss:.4f} | case_f1={test_m['macro_f1']:.4f} "
              f"| accuracy={test_m['accuracy']:.4f} | cases={test_m['cases']}")

if __name__=='__main__':
    main()
