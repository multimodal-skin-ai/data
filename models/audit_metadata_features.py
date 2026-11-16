"""SCIN metadata quality and predictiveness audit, independent of image / torch.

Run from /content/data after uploading to models/:
python models/audit_metadata_features.py --csv '/content/drive/MyDrive/데이터사이언스응용 - 고양이 팀/integration/multimodal_integrated_dataset.csv' --out-dir '/content/drive/MyDrive/데이터사이언스응용 - 고양이 팀/outputs/metadata_audit'

No test labels are used. The logistic regression comparison is a diagnostic
hard-label baseline, NOT a replacement for the previous soft-target MLP.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif

FEATURE_GROUPS = {
    'textures': ['textures_raised_or_bumpy', 'textures_flat', 'textures_rough_or_flaky', 'textures_fluid_filled', 'textures_has_recorded_yes'],
    'body_parts': ['body_parts_head_or_neck', 'body_parts_arm', 'body_parts_palm', 'body_parts_back_of_hand', 'body_parts_torso_front', 'body_parts_torso_back', 'body_parts_genitalia_or_groin', 'body_parts_buttocks', 'body_parts_leg', 'body_parts_foot_top_or_side', 'body_parts_foot_sole', 'body_parts_other', 'body_parts_has_recorded_yes'],
    'symptoms': ['condition_symptoms_bleeding', 'condition_symptoms_increasing_size', 'condition_symptoms_darkening', 'condition_symptoms_itching', 'condition_symptoms_burning', 'condition_symptoms_pain', 'condition_symptoms_has_recorded_yes'],
    'duration': ['duration_LESS_THAN_ONE_WEEK', 'duration_MORE_THAN_FIVE_YEARS', 'duration_MORE_THAN_ONE_YEAR', 'duration_ONE_DAY', 'duration_ONE_TO_FOUR_WEEKS', 'duration_ONE_TO_THREE_MONTHS', 'duration_SINCE_CHILDHOOD', 'duration_THREE_TO_TWELVE_MONTHS', 'duration_UNKNOWN'],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
assert len(FEATURES) == 34


def load_cases(path):
    df = pd.read_csv(path, dtype={'case_id': str}, low_memory=False)
    needed = {'case_id', 'split', 'soft_label', *FEATURES}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df['split'] = df['split'].astype(str).str.strip().str.lower()
    if df['case_id'].isna().any() or df['split'].isna().any():
        raise ValueError('case_id/split contains missing values')
    if df.groupby('case_id')['split'].nunique().gt(1).any():
        raise ValueError('CASE LEAKAGE: case_id occurs in multiple splits')
    inconsistent = []
    for col in ['soft_label', *FEATURES]:
        conflict = df.groupby('case_id')[col].nunique(dropna=False).gt(1)
        if conflict.any():
            inconsistent.append((col, int(conflict.sum())))
    if inconsistent:
        raise ValueError(f'Conflicting values within same case: {inconsistent}')
    split_images = df['split'].value_counts().to_dict()
    cases = df.drop_duplicates('case_id').reset_index(drop=True)
    split_cases = cases['split'].value_counts().to_dict()
    print('Image rows by split:', split_images)
    print('Unique cases by split:', split_cases)
    if not {'train', 'validation'}.issubset(set(cases.split)):
        raise ValueError('Train and validation cases are required')
    return cases


def y_from_soft(series):
    ys = []
    for raw in series:
        arr = np.asarray(ast.literal_eval(raw) if isinstance(raw, str) else raw, dtype=float)
        if arr.shape != (33,) or not np.isfinite(arr).all() or (arr < 0).any() or arr.sum() <= 0:
            raise ValueError(f'Invalid soft label shape/value: {arr.shape}')
        ys.append(int(arr.argmax()))
    return np.asarray(ys, dtype=int)


def to_numeric(df):
    return df[FEATURES].apply(pd.to_numeric, errors='coerce')


def feature_quality(train, val, xtrain, xval):
    records = []
    for group, columns in FEATURE_GROUPS.items():
        for f in columns:
            t, v = xtrain[f], xval[f]
            observed = t.dropna()
            unique = sorted(observed.unique().tolist())
            not_binary = any(z not in (0, 1) for z in unique)
            records.append({
                'group': group, 'feature': f,
                'train_missing_n': int(t.isna().sum()), 'train_missing_pct': float(t.isna().mean()*100),
                'val_missing_n': int(v.isna().sum()), 'val_missing_pct': float(v.isna().mean()*100),
                'train_one_n': int((t == 1).sum()), 'train_one_pct': float((t == 1).mean()*100),
                'val_one_n': int((v == 1).sum()), 'val_one_pct': float((v == 1).mean()*100),
                'positive_rate_gap_pp': float(((v == 1).mean()-(t == 1).mean())*100),
                'train_unique_count': int(observed.nunique()),
                'train_constant': bool(observed.nunique() <= 1),
                'non_binary_values': bool(not_binary),
            })
    return pd.DataFrame(records)


def pattern_checks(train, val, xtrain, xval, yt, yv):
    # Missingness is represented by a distinct marker (-99) rather than zero.
    xt = xtrain.fillna(-99)
    xv = xval.fillna(-99)
    tp = [tuple(row) for row in xt.to_numpy()]
    vp = [tuple(row) for row in xv.to_numpy()]
    groups = {}
    for p, y in zip(tp, yt):
        groups.setdefault(p, Counter())[int(y)] += 1
    duplicates = {p: cnt for p, cnt in groups.items() if sum(cnt.values()) > 1}
    conflicting = {p: cnt for p, cnt in groups.items() if len(cnt) > 1}
    seen = set(groups)
    val_unseen = sum(p not in seen for p in vp)
    print('\n=== 34-feature pattern diagnostics (TRAIN only for label comparisons) ===')
    print(f'Unique train input patterns: {len(groups)} / {len(tp)} cases')
    print(f'Train patterns shared by >=2 cases: {len(duplicates)}')
    print(f'Train patterns with conflicting hard labels: {len(conflicting)}')
    print(f'Cases in conflicting train patterns: {sum(sum(c.values()) for c in conflicting.values())}')
    print(f'Validation cases with unseen input pattern: {val_unseen}/{len(vp)}')
    # Distinct input patterns and contradictory labels limit deterministic classification,
    # but TRAIN pattern upper bounds do not establish generalization performance.
    if groups:
        memorization_train_accuracy = sum(max(c.values()) for c in groups.values()) / len(tp)
        print(f'Train memorization majority-label accuracy by exact pattern: {memorization_train_accuracy:.4f} (not a validation bound)')
    return pd.DataFrame([{
        'metric': 'unique_train_patterns', 'value': len(groups)},
        {'metric':'train_patterns_with_duplicates', 'value':len(duplicates)},
        {'metric':'train_patterns_with_conflicting_labels', 'value':len(conflicting)},
        {'metric':'train_cases_in_conflicting_patterns', 'value':sum(sum(c.values()) for c in conflicting.values())},
        {'metric':'validation_cases_with_unseen_pattern','value':val_unseen}
    ])


def group_baselines(xtrain, xval, yt, yv, random_state=42):
    combos = {
        'all_34': FEATURES,
        'body_plus_symptoms': FEATURE_GROUPS['body_parts'] + FEATURE_GROUPS['symptoms'],
        **FEATURE_GROUPS,
    }
    records = []
    labels = list(range(33))
    for group, cols in combos.items():
        for weighting in ('none', 'balanced'):
            # Fit imputation on TRAIN ONLY; no image, no validation tuning, no test access.
            estimator = make_pipeline(
                SimpleImputer(strategy='constant', fill_value=-1, keep_empty_features=True),
                LogisticRegression(max_iter=1200, C=1.0, class_weight=None if weighting=='none' else 'balanced', random_state=random_state),
            )
            estimator.fit(xtrain[cols], yt)
            pred = estimator.predict(xval[cols])
            precision, recall, f1, support = precision_recall_fscore_support(yv, pred, labels=labels, zero_division=0)
            macro = float(f1_score(yv, pred, labels=labels, average='macro', zero_division=0))
            acc = float(accuracy_score(yv, pred))
            records.append({'features':group,'num_features':len(cols),'class_weight':weighting,
                            'case_macro_f1':macro,'case_accuracy':acc, 'predicted_classes':len(np.unique(pred)),
                            'classes_with_f1_gt_zero':int((f1>0).sum())})
            print(f'{group:20} | {weighting:8} | F1={macro:.4f} | Acc={acc:.4f} | predicted classes={len(np.unique(pred))}')
    return pd.DataFrame(records).sort_values('case_macro_f1',ascending=False)


def main():
    parser = argparse.ArgumentParser(description='SCIN 34-feature quality / predictive signal audit')
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    cases=load_cases(args.csv)
    train=cases[cases['split']=='train'].copy()
    val=cases[cases['split']=='validation'].copy()
    xt,xv=to_numeric(train),to_numeric(val)
    yt,yv=y_from_soft(train.soft_label),y_from_soft(val.soft_label)
    print('\n=== Feature quality ===')
    quality=feature_quality(train,val,xt,xv)
    print('Constant features:',int(quality.train_constant.sum()), '/ 34')
    print('Features >95% zero in TRAIN:',int((quality.train_one_pct<5).sum()), '/ 34')
    print('Features with missing >10% in TRAIN:',int((quality.train_missing_pct>10).sum()), '/ 34')
    print('Non-binary features:',int(quality.non_binary_values.sum()))
    print('\nSparsest and most-shifted features:')
    print(quality.sort_values('train_one_pct').head(10)[['group','feature','train_one_pct','val_one_pct','train_missing_pct']].to_string(index=False))
    print('\nLargest train-validation positive rate gaps:')
    print(quality.assign(abs_gap=lambda x:x.positive_rate_gap_pp.abs()).sort_values('abs_gap',ascending=False).head(10)[['feature','train_one_pct','val_one_pct','positive_rate_gap_pp']].to_string(index=False))
    quality.to_csv(args.out_dir/'feature_quality.csv',index=False,encoding='utf-8-sig')
    patt=pattern_checks(train,val,xt,xv,yt,yv)
    patt.to_csv(args.out_dir/'pattern_summary.csv',index=False,encoding='utf-8-sig')
    print('\n=== TRAIN-only univariate feature mutual information (exploratory, biased for small samples) ===')
    imputer=SimpleImputer(strategy='constant',fill_value=-1,keep_empty_features=True)
    xmi=imputer.fit_transform(xt)
    mi=mutual_info_classif(xmi,yt,discrete_features=True,random_state=args.seed)
    mi_table=pd.DataFrame({'feature':FEATURES,'mutual_info_nats':mi})
    mi_table['group'] = mi_table.feature.map({f:g for g,fs in FEATURE_GROUPS.items() for f in fs})
    mi_table.sort_values('mutual_info_nats',ascending=False).to_csv(args.out_dir/'mutual_info_train.csv',index=False,encoding='utf-8-sig')
    print(mi_table.sort_values('mutual_info_nats',ascending=False).head(12).to_string(index=False))
    print('\n=== Diagnostic logistic regression (hard-label baseline; 33-class Macro F1) ===')
    scores=group_baselines(xt,xv,yt,yv,args.seed)
    scores.to_csv(args.out_dir/'group_baselines.csv',index=False,encoding='utf-8-sig')
    print('\n=== Sorted baselines ===')
    print(scores.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    print('\nSaved 4 CSV reports to:',args.out_dir)
    print('TEST split was not used. Compare these as diagnostic baselines, not as direct replacements for soft-label MLP.')

if __name__=='__main__':
    main()
