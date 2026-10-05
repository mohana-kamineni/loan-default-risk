"""
Modeling and Validation Engine
Provides leakage-free cross-validation, identical fold reuse, Model A vs Model B
paired evaluation, solvent-at-window-end subcohort analysis, and Out-of-Time evaluation.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

RANDOM_SEED = 42

STATIC_COLS = [
    'feat_loan_amount', 'feat_duration_months', 'feat_monthly_payment',
    'feat_payment_to_amount_ratio', 'feat_account_age_days',
    'feat_client_age_years', 'feat_client_is_female',
    'feat_district_salary', 'feat_district_unemployment',
    'feat_district_crimes', 'feat_district_entrepreneurs'
]

PRE_LOAN_COLS = [
    'feat_had_pre_loan_overdraft', 'feat_pre_loan_tx_count'
]

MODEL_A_COLS = STATIC_COLS + PRE_LOAN_COLS  # 13 features


def get_model_b_cols(window_days: int) -> List[str]:
    """Return all 26 features for Model B (13 Model A + 13 Window Behavioural)."""
    w_cols = [
        f'feat_w{window_days}_trans_count',
        f'feat_w{window_days}_inflow_total',
        f'feat_w{window_days}_outflow_total',
        f'feat_w{window_days}_net_cash_flow',
        f'feat_w{window_days}_outflow_to_inflow_ratio',
        f'feat_w{window_days}_cash_withdrawal_amount',
        f'feat_w{window_days}_cash_withdrawal_ratio',
        f'feat_w{window_days}_balance_mean',
        f'feat_w{window_days}_balance_min',
        f'feat_w{window_days}_balance_slope',
        f'feat_w{window_days}_had_negative_balance',
        f'feat_w{window_days}_had_sanction_interest',
        f'feat_w{window_days}_min_bal_to_payment_ratio'
    ]
    return MODEL_A_COLS + w_cols


def create_and_persist_splits(df: pd.DataFrame, target_col: str, filepath: str,
                                n_splits: int = 5, n_repeats: int = 5) -> List[Dict[str, Any]]:
    """
    Generate repeated stratified K-fold splits ONCE and persist to disk.
    Ensures identical folds across all windows, models, and ablations.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_SEED)
    splits = []
    records = []

    for split_id, (train_idx, test_idx) in enumerate(rskf.split(df, df[target_col])):
        rep = split_id // n_splits
        fold = split_id % n_splits
        splits.append({
            'split_id': split_id,
            'repeat': rep,
            'fold': fold,
            'train_idx': train_idx,
            'test_idx': test_idx
        })
        for idx in test_idx:
            records.append({
                'split_id': split_id,
                'repeat': rep,
                'fold': fold,
                'loan_id': int(df.iloc[idx]['loan_id']),
                'is_test': 1
            })

    # Save to disk
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame(records).to_csv(filepath, index=False)
    return splits


def get_model_instance(model_name: str, random_state: int = RANDOM_SEED):
    """Instantiate model pipeline with strict leak-free tuning."""
    if model_name == 'Logistic Regression':
        # L2 regularized logistic regression with inner CV selection of C on train fold
        clf = LogisticRegressionCV(
            Cs=[0.01, 0.1, 1.0, 10.0],
            cv=5,
            scoring='average_precision',
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        )
        return Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    elif model_name == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=random_state,
            n_jobs=-1
        )
    elif model_name == 'HistGradientBoosting':
        return HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_cv_experiment(df: pd.DataFrame, splits: List[Dict[str, Any]],
                       model_name: str, window_days: int,
                       target_col: str = 'target_completed') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run paired Model A vs Model B evaluation across all 25 splits.
    Returns:
    1. fold_metrics_df: metrics per split (PR-AUC, ROC-AUC, Brier, Deltas, full & solvent).
    2. predictions_df: out-of-fold prediction probabilities.
    """
    feature_a_cols = MODEL_A_COLS
    feature_b_cols = get_model_b_cols(window_days)

    fold_metrics = []
    predictions = []

    for s in splits:
        split_id = s['split_id']
        rep = s['repeat']
        fold = s['fold']
        train_idx = s['train_idx']
        test_idx = s['test_idx']

        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]

        y_train = train_data[target_col].values.astype(int)
        y_test = test_data[target_col].values.astype(int)

        n_test = len(y_test)
        n_pos_test = int(np.sum(y_test))
        n_neg_test = n_test - n_pos_test

        # --- MODEL A (Historical Baseline) ---
        X_train_a = train_data[feature_a_cols].values
        X_test_a = test_data[feature_a_cols].values

        clf_a = get_model_instance(model_name, random_state=RANDOM_SEED + split_id)
        clf_a.fit(X_train_a, y_train)
        pred_a_test = clf_a.predict_proba(X_test_a)[:, 1]

        prauc_a = average_precision_score(y_test, pred_a_test)
        rocauc_a = roc_auc_score(y_test, pred_a_test)
        brier_a = brier_score_loss(y_test, pred_a_test)

        # --- MODEL B (Behavioural Model) ---
        X_train_b = train_data[feature_b_cols].values
        X_test_b = test_data[feature_b_cols].values

        clf_b = get_model_instance(model_name, random_state=RANDOM_SEED + split_id)
        clf_b.fit(X_train_b, y_train)
        pred_b_test = clf_b.predict_proba(X_test_b)[:, 1]

        prauc_b = average_precision_score(y_test, pred_b_test)
        rocauc_b = roc_auc_score(y_test, pred_b_test)
        brier_b = brier_score_loss(y_test, pred_b_test)

        # Paired Deltas (Model B - Model A)
        delta_prauc = prauc_b - prauc_a
        delta_rocauc = rocauc_b - rocauc_a
        delta_brier = brier_b - brier_a  # Negative delta indicates improvement

        # --- SOLVENT-AT-WINDOW-END SUBCONHORT ANALYSIS ---
        # Identify test loans that had NO negative balance in window
        had_neg_col = f'feat_w{window_days}_had_negative_balance'
        solvent_mask = (test_data[had_neg_col].values == 0)
        n_solvent_test = int(np.sum(solvent_mask))
        n_pos_solvent_test = int(np.sum(y_test[solvent_mask]))

        if n_pos_solvent_test > 0 and (n_solvent_test - n_pos_solvent_test) > 0:
            prauc_a_solv = average_precision_score(y_test[solvent_mask], pred_a_test[solvent_mask])
            prauc_b_solv = average_precision_score(y_test[solvent_mask], pred_b_test[solvent_mask])
            delta_prauc_solv = prauc_b_solv - prauc_a_solv

            rocauc_a_solv = roc_auc_score(y_test[solvent_mask], pred_a_test[solvent_mask])
            rocauc_b_solv = roc_auc_score(y_test[solvent_mask], pred_b_test[solvent_mask])
            delta_rocauc_solv = rocauc_b_solv - rocauc_a_solv

            brier_a_solv = brier_score_loss(y_test[solvent_mask], pred_a_test[solvent_mask])
            brier_b_solv = brier_score_loss(y_test[solvent_mask], pred_b_test[solvent_mask])
        else:
            prauc_a_solv = np.nan
            prauc_b_solv = np.nan
            delta_prauc_solv = np.nan
            rocauc_a_solv = np.nan
            rocauc_b_solv = np.nan
            delta_rocauc_solv = np.nan
            brier_a_solv = np.nan
            brier_b_solv = np.nan

        fold_metrics.append({
            'split_id': split_id,
            'repeat': rep,
            'fold': fold,
            'model': model_name,
            'window_days': window_days,
            'n_test': n_test,
            'n_pos_test': n_pos_test,
            'n_neg_test': n_neg_test,
            # Model A full
            'prauc_a': prauc_a,
            'rocauc_a': rocauc_a,
            'brier_a': brier_a,
            # Model B full
            'prauc_b': prauc_b,
            'rocauc_b': rocauc_b,
            'brier_b': brier_b,
            # Deltas full
            'delta_prauc': delta_prauc,
            'delta_rocauc': delta_rocauc,
            'delta_brier': delta_brier,
            # Solvent subset metrics
            'n_solvent_test': n_solvent_test,
            'n_pos_solvent_test': n_pos_solvent_test,
            'prauc_a_solvent': prauc_a_solv,
            'prauc_b_solvent': prauc_b_solv,
            'delta_prauc_solvent': delta_prauc_solv,
            'rocauc_a_solvent': rocauc_a_solv,
            'rocauc_b_solvent': rocauc_b_solv,
            'delta_rocauc_solvent': delta_rocauc_solv,
            'brier_a_solvent': brier_a_solv,
            'brier_b_solvent': brier_b_solv
        })

        for i, idx in enumerate(test_idx):
            predictions.append({
                'split_id': split_id,
                'repeat': rep,
                'fold': fold,
                'loan_id': int(test_data.iloc[i]['loan_id']),
                'y_true': int(y_test[i]),
                'prob_model_a': float(pred_a_test[i]),
                'prob_model_b': float(pred_b_test[i]),
                'is_solvent': int(solvent_mask[i]),
                'window_days': window_days,
                'model': model_name
            })

    return pd.DataFrame(fold_metrics), pd.DataFrame(predictions)


def run_oot_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        model_name: str, window_days: int,
                        target_col: str = 'target_completed') -> Dict[str, Any]:
    """
    Execute Out-of-Time evaluation:
    Train: 1993-1995 (strictly fit preprocessing & model)
    Test: 1996-1997
    """
    feature_a_cols = MODEL_A_COLS
    feature_b_cols = get_model_b_cols(window_days)

    y_train = train_df[target_col].values.astype(int)
    y_test = test_df[target_col].values.astype(int)

    n_train = len(y_train)
    n_pos_train = int(np.sum(y_train))
    n_test = len(y_test)
    n_pos_test = int(np.sum(y_test))

    # Model A
    X_train_a = train_df[feature_a_cols].values
    X_test_a = test_df[feature_a_cols].values

    clf_a = get_model_instance(model_name, random_state=RANDOM_SEED)
    clf_a.fit(X_train_a, y_train)
    pred_a_test = clf_a.predict_proba(X_test_a)[:, 1]

    prauc_a = average_precision_score(y_test, pred_a_test)
    rocauc_a = roc_auc_score(y_test, pred_a_test)
    brier_a = brier_score_loss(y_test, pred_a_test)

    # Model B
    X_train_b = train_df[feature_b_cols].values
    X_test_b = test_df[feature_b_cols].values

    clf_b = get_model_instance(model_name, random_state=RANDOM_SEED)
    clf_b.fit(X_train_b, y_train)
    pred_b_test = clf_b.predict_proba(X_test_b)[:, 1]

    prauc_b = average_precision_score(y_test, pred_b_test)
    rocauc_b = roc_auc_score(y_test, pred_b_test)
    brier_b = brier_score_loss(y_test, pred_b_test)

    # Solvent subset in OOT test
    had_neg_col = f'feat_w{window_days}_had_negative_balance'
    solvent_mask = (test_df[had_neg_col].values == 0)
    n_solv = int(np.sum(solvent_mask))
    n_pos_solv = int(np.sum(y_test[solvent_mask]))

    if n_pos_solv > 0 and (n_solv - n_pos_solv) > 0:
        prauc_a_solv = average_precision_score(y_test[solvent_mask], pred_a_test[solvent_mask])
        prauc_b_solv = average_precision_score(y_test[solvent_mask], pred_b_test[solvent_mask])
        rocauc_a_solv = roc_auc_score(y_test[solvent_mask], pred_a_test[solvent_mask])
        rocauc_b_solv = roc_auc_score(y_test[solvent_mask], pred_b_test[solvent_mask])
    else:
        prauc_a_solv = np.nan
        prauc_b_solv = np.nan
        rocauc_a_solv = np.nan
        rocauc_b_solv = np.nan

    return {
        'model': model_name,
        'window_days': window_days,
        'n_train': n_train,
        'n_pos_train': n_pos_train,
        'n_test': n_test,
        'n_pos_test': n_pos_test,
        'prauc_a': prauc_a,
        'prauc_b': prauc_b,
        'delta_prauc': prauc_b - prauc_a,
        'rocauc_a': rocauc_a,
        'rocauc_b': rocauc_b,
        'delta_rocauc': rocauc_b - rocauc_a,
        'brier_a': brier_a,
        'brier_b': brier_b,
        'delta_brier': brier_b - brier_a,
        'n_solvent_test': n_solv,
        'n_pos_solvent_test': n_pos_solv,
        'prauc_a_solvent': prauc_a_solv,
        'prauc_b_solvent': prauc_b_solv,
        'delta_prauc_solvent': prauc_b_solv - prauc_a_solv,
        'rocauc_a_solvent': rocauc_a_solv,
        'rocauc_b_solvent': rocauc_b_solv,
        'delta_rocauc_solvent': rocauc_b_solv - rocauc_a_solv
    }


def compute_summary_statistics(deltas: np.ndarray) -> Dict[str, float]:
    """Compute mean, median, std, win rate, and Wilcoxon p-value for paired deltas."""
    valid_d = deltas[~np.isnan(deltas)]
    if len(valid_d) == 0:
        return {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'win_rate': np.nan, 'wilcoxon_p': np.nan}
    
    mean_d = float(np.mean(valid_d))
    median_d = float(np.median(valid_d))
    std_d = float(np.std(valid_d, ddof=1)) if len(valid_d) > 1 else 0.0
    win_rate = float(np.mean(valid_d > 0))

    # Wilcoxon signed-rank test
    # Note: requires non-zero differences
    diffs_nonzero = valid_d[valid_d != 0]
    if len(diffs_nonzero) >= 5:
        try:
            stat, p_val = stats.wilcoxon(diffs_nonzero)
            p_val = float(p_val)
        except Exception:
            p_val = np.nan
    else:
        p_val = np.nan

    return {
        'mean': mean_d,
        'median': median_d,
        'std': std_d,
        'win_rate': win_rate,
        'wilcoxon_p': p_val
    }
