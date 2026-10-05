"""
Machine Learning Experiment Runner
Executes the full experiment matrix across Primary (Completed N=234) and
Secondary (Expanded N=656) cohorts, 30/60/90-day windows, Model A vs B ablations,
solvent subcohort analysis, and Out-of-Time evaluation.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from datetime import datetime

from src.modeling import (
    create_and_persist_splits, run_cv_experiment, run_oot_experiment,
    compute_summary_statistics, MODEL_A_COLS, get_model_b_cols, get_model_instance
)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def main():
    print("=================================================================")
    print("MACHINE LEARNING EXPERIMENTS: MODELING & VALIDATION")
    print("=================================================================\n")

    # 1. Load Data
    print("--- 1. Loading Processed Datasets ---")
    df_30 = pd.read_csv(os.path.join(PROCESSED_DIR, 'completed_cohort_w30d.csv'))
    df_60 = pd.read_csv(os.path.join(PROCESSED_DIR, 'completed_cohort_w60d.csv'))
    df_90 = pd.read_csv(os.path.join(PROCESSED_DIR, 'completed_cohort_w90d.csv'))
    df_exp90 = pd.read_csv(os.path.join(PROCESSED_DIR, 'expanded_cohort_w90d.csv'))

    df_30['loan_date'] = pd.to_datetime(df_30['loan_date'])
    df_60['loan_date'] = pd.to_datetime(df_60['loan_date'])
    df_90['loan_date'] = pd.to_datetime(df_90['loan_date'])
    df_exp90['loan_date'] = pd.to_datetime(df_exp90['loan_date'])

    print(f"Primary Cohort 30d: {df_30.shape}")
    print(f"Primary Cohort 60d: {df_60.shape}")
    print(f"Primary Cohort 90d: {df_90.shape}")
    print(f"Expanded Cohort 90d: {df_exp90.shape}")

    # 2. Generate and Persist CV Folds ONCE for Primary Completed Cohort
    print("\n--- 2. Generating & Persisting Identical CV Splits ---")
    splits_primary_path = os.path.join(PROCESSED_DIR, 'cv_splits_completed_n234.csv')
    splits_primary = create_and_persist_splits(df_90, target_col='target_completed', filepath=splits_primary_path)
    print(f"Primary cohort splits: 5 folds x 5 repeats = {len(splits_primary)} units saved to {splits_primary_path}")

    # Verify fold distribution
    test_pos_counts = [int(df_90.iloc[s['test_idx']]['target_completed'].sum()) for s in splits_primary]
    print(f"Default cases per test fold: min={min(test_pos_counts)}, max={max(test_pos_counts)}, median={np.median(test_pos_counts)}")

    # Generate splits for Expanded Cohort (N=656)
    splits_exp_path = os.path.join(PROCESSED_DIR, 'cv_splits_expanded_n656.csv')
    splits_expanded = create_and_persist_splits(df_exp90, target_col='target_expanded', filepath=splits_exp_path)
    print(f"Expanded cohort splits: 5 folds x 5 repeats = {len(splits_expanded)} units saved to {splits_exp_path}")

    models = ['Logistic Regression', 'Random Forest', 'HistGradientBoosting']
    windows = [30, 60, 90]
    data_by_w = {30: df_30, 60: df_60, 90: df_90}

    all_cv_results = []
    all_predictions = []

    # 3. Execute Primary Completed Cohort Experiments across Models and Windows
    print("\n--- 3. Running Primary Completed Cohort (N=234) CV Experiments ---")
    for model_name in models:
        for w in windows:
            print(f"Running {model_name:<22} | Window {w:2d}d...")
            df_w = data_by_w[w]
            metrics_df, preds_df = run_cv_experiment(
                df=df_w,
                splits=splits_primary,
                model_name=model_name,
                window_days=w,
                target_col='target_completed'
            )
            metrics_df['cohort'] = 'Primary (Completed N=234)'
            preds_df['cohort'] = 'Primary (Completed N=234)'
            all_cv_results.append(metrics_df)
            all_predictions.append(preds_df)

    # 4. Execute Secondary Sensitivity Analysis on Expanded Cohort (N=656) at 90d
    print("\n--- 4. Running Secondary Expanded Cohort (N=656) 90d Sensitivity ---")
    for model_name in models:
        print(f"Running Expanded {model_name:<13} | Window 90d...")
        metrics_df, preds_df = run_cv_experiment(
            df=df_exp90,
            splits=splits_expanded,
            model_name=model_name,
            window_days=90,
            target_col='target_expanded'
        )
        metrics_df['cohort'] = 'Secondary (Expanded N=656)'
        preds_df['cohort'] = 'Secondary (Expanded N=656)'
        all_cv_results.append(metrics_df)
        all_predictions.append(preds_df)

    # Combine and save all CV results
    cv_master_df = pd.concat(all_cv_results, ignore_index=True)
    preds_master_df = pd.concat(all_predictions, ignore_index=True)

    cv_master_path = os.path.join(EXPERIMENTS_DIR, 'cv_fold_metrics_master.csv')
    cv_master_df.to_csv(cv_master_path, index=False)
    print(f"\nSaved all fold metrics to {cv_master_path}")

    preds_master_path = os.path.join(EXPERIMENTS_DIR, 'predictions_master.csv')
    preds_master_df.to_csv(preds_master_path, index=False)
    print(f"Saved all predictions to {preds_master_path}")

    # 5. Out-of-Time (OOT) Chronological Split Experiments
    print("\n--- 5. Running Chronological Out-of-Time (OOT) Experiments ---")
    # Primary completed cohort: Train 1993-1995, Test 1996-1997
    oot_results = []
    for w in windows:
        df_w = data_by_w[w]
        train_df = df_w[df_w['loan_date'].dt.year <= 1995].copy()
        test_df = df_w[df_w['loan_date'].dt.year >= 1996].copy()

        assert len(train_df) == 156, f"Expected 156 train loans, got {len(train_df)}"
        assert len(test_df) == 78, f"Expected 78 test loans, got {len(test_df)}"
        assert int(train_df['target_completed'].sum()) == 22, "Expected 22 train defaults"
        assert int(test_df['target_completed'].sum()) == 9, "Expected 9 test defaults"

        for model_name in models:
            res = run_oot_experiment(train_df, test_df, model_name=model_name, window_days=w)
            res['cohort'] = 'Primary (Completed N=234)'
            oot_results.append(res)

    oot_df = pd.DataFrame(oot_results)
    oot_path = os.path.join(EXPERIMENTS_DIR, 'oot_metrics.csv')
    oot_df.to_csv(oot_path, index=False)
    print(f"Saved OOT metrics to {oot_path}")

    # 6. Synthesize Summary Tables
    print("\n=================================================================")
    print("RESULTS SYNTHESIS & REPORTING")
    print("=================================================================")

    # Table 1: Model A vs Model B Performance across Models and Windows (Primary Cohort)
    prim_cv = cv_master_df[cv_master_df['cohort'] == 'Primary (Completed N=234)']
    summary_rows = []

    for (model, w), group in prim_cv.groupby(['model', 'window_days']):
        # Full cohort
        prauc_a_m, prauc_a_s = group['prauc_a'].mean(), group['prauc_a'].std()
        prauc_b_m, prauc_b_s = group['prauc_b'].mean(), group['prauc_b'].std()
        rocauc_a_m, rocauc_a_s = group['rocauc_a'].mean(), group['rocauc_a'].std()
        rocauc_b_m, rocauc_b_s = group['rocauc_b'].mean(), group['rocauc_b'].std()
        brier_a_m, brier_a_s = group['brier_a'].mean(), group['brier_a'].std()
        brier_b_m, brier_b_s = group['brier_b'].mean(), group['brier_b'].std()

        delta_stats = compute_summary_statistics(group['delta_prauc'].values)

        # Solvent subset
        solv_group = group.dropna(subset=['prauc_b_solvent', 'prauc_a_solvent'])
        solv_a_m = solv_group['prauc_a_solvent'].mean()
        solv_b_m = solv_group['prauc_b_solvent'].mean()
        solv_delta = compute_summary_statistics(solv_group['delta_prauc_solvent'].values)

        summary_rows.append({
            'Model': model,
            'Window': f'{w}d',
            'Model A PR-AUC': f"{prauc_a_m:.3f} +/- {prauc_a_s:.3f}",
            'Model B PR-AUC': f"{prauc_b_m:.3f} +/- {prauc_b_s:.3f}",
            'Delta Mean': delta_stats['mean'],
            'Delta Median': delta_stats['median'],
            'Delta Std': delta_stats['std'],
            'Win Rate': f"{delta_stats['win_rate']*100:.1f}%",
            'Wilcoxon p': delta_stats['wilcoxon_p'],
            'Model A ROC-AUC': f"{rocauc_a_m:.3f} +/- {rocauc_a_s:.3f}",
            'Model B ROC-AUC': f"{rocauc_b_m:.3f} +/- {rocauc_b_s:.3f}",
            'Model A Brier': f"{brier_a_m:.3f} +/- {brier_a_s:.3f}",
            'Model B Brier': f"{brier_b_m:.3f} +/- {brier_b_s:.3f}",
            'Solvent Model A PR-AUC': f"{solv_a_m:.3f}",
            'Solvent Model B PR-AUC': f"{solv_b_m:.3f}",
            'Solvent Delta Mean': solv_delta['mean'],
            'Solvent Win Rate': f"{solv_delta['win_rate']*100:.1f}%"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(EXPERIMENTS_DIR, 'summary_model_a_b_comparison.csv')
    summary_df.to_csv(summary_path, index=False)
    print("\n--- Summary Table 1: Model A vs Model B (Primary Completed N=234) ---")
    print(summary_df[['Model', 'Window', 'Model A PR-AUC', 'Model B PR-AUC', 'Delta Mean', 'Win Rate', 'Wilcoxon p', 'Solvent Delta Mean']])

    # Table 2: Paired Window Comparisons (30d vs 60d vs 90d for Model B)
    # Because folds are identical, we merge splits for 30d, 60d, 90d
    win_comp_rows = []
    for model in models:
        m_sub = prim_cv[prim_cv['model'] == model]
        m30 = m_sub[m_sub['window_days'] == 30].set_index('split_id')['prauc_b']
        m60 = m_sub[m_sub['window_days'] == 60].set_index('split_id')['prauc_b']
        m90 = m_sub[m_sub['window_days'] == 90].set_index('split_id')['prauc_b']

        d_60_30 = compute_summary_statistics((m60 - m30).values)
        d_90_60 = compute_summary_statistics((m90 - m60).values)
        d_90_30 = compute_summary_statistics((m90 - m30).values)

        win_comp_rows.append({
            'Model': model,
            'Comparison': '60d vs 30d',
            'Mean Delta': d_60_30['mean'],
            'Median Delta': d_60_30['median'],
            'Std Delta': d_60_30['std'],
            'Win Rate (Later > Earlier)': f"{d_60_30['win_rate']*100:.1f}%",
            'Wilcoxon p': d_60_30['wilcoxon_p']
        })
        win_comp_rows.append({
            'Model': model,
            'Comparison': '90d vs 60d',
            'Mean Delta': d_90_60['mean'],
            'Median Delta': d_90_60['median'],
            'Std Delta': d_90_60['std'],
            'Win Rate (Later > Earlier)': f"{d_90_60['win_rate']*100:.1f}%",
            'Wilcoxon p': d_90_60['wilcoxon_p']
        })
        win_comp_rows.append({
            'Model': model,
            'Comparison': '90d vs 30d',
            'Mean Delta': d_90_30['mean'],
            'Median Delta': d_90_30['median'],
            'Std Delta': d_90_30['std'],
            'Win Rate (Later > Earlier)': f"{d_90_30['win_rate']*100:.1f}%",
            'Wilcoxon p': d_90_30['wilcoxon_p']
        })

    win_comp_df = pd.DataFrame(win_comp_rows)
    win_comp_path = os.path.join(EXPERIMENTS_DIR, 'summary_window_comparisons.csv')
    win_comp_df.to_csv(win_comp_path, index=False)
    print("\n--- Summary Table 2: Paired Information Progression (30d vs 60d vs 90d) ---")
    print(win_comp_df[['Model', 'Comparison', 'Mean Delta', 'Median Delta', 'Win Rate (Later > Earlier)', 'Wilcoxon p']])

    # Table 3: OOT Summary
    print("\n--- Summary Table 3: Chronological Out-of-Time Validation (Train: 93-95 [22 def], Test: 96-97 [9 def]) ---")
    print(oot_df[['model', 'window_days', 'prauc_a', 'prauc_b', 'delta_prauc', 'rocauc_a', 'rocauc_b', 'delta_rocauc', 'prauc_b_solvent']])

    # Table 4: Sensitivity on Expanded Cohort (N=656) at 90d
    exp_cv = cv_master_df[cv_master_df['cohort'] == 'Secondary (Expanded N=656)']
    exp_rows = []
    for model, group in exp_cv.groupby('model'):
        d_stats = compute_summary_statistics(group['delta_prauc'].values)
        exp_rows.append({
            'Model': model,
            'Expanded Model A PR-AUC': f"{group['prauc_a'].mean():.3f} +/- {group['prauc_a'].std():.3f}",
            'Expanded Model B PR-AUC': f"{group['prauc_b'].mean():.3f} +/- {group['prauc_b'].std():.3f}",
            'Expanded Delta Mean': d_stats['mean'],
            'Expanded Win Rate': f"{d_stats['win_rate']*100:.1f}%",
            'Expanded Wilcoxon p': d_stats['wilcoxon_p'],
            'Expanded Model A ROC-AUC': f"{group['rocauc_a'].mean():.3f} +/- {group['rocauc_a'].std():.3f}",
            'Expanded Model B ROC-AUC': f"{group['rocauc_b'].mean():.3f} +/- {group['rocauc_b'].std():.3f}"
        })
    exp_summary_df = pd.DataFrame(exp_rows)
    exp_summary_path = os.path.join(EXPERIMENTS_DIR, 'summary_expanded_sensitivity_90d.csv')
    exp_summary_df.to_csv(exp_summary_path, index=False)
    print("\n--- Summary Table 4: Secondary Sensitivity Cohort (Expanded N=656, 75 def) at 90d ---")
    print(exp_summary_df)

    print("\nExperiments completed successfully.")


if __name__ == '__main__':
    main()
