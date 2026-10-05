"""
Pipeline Orchestrator: Stages 1 to 4
Runs ingestion, cohort creation, feature extraction, and saves processed datasets.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.ingestion import BerkaDataLoader
from src.cohorts import assign_cohorts_and_eligibility, get_cohort_summary
from src.features import FeatureExtractor
from src.data_dictionary import get_data_dictionary_df

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("=== RUNNING DATASET PIPELINE ===")

    # 1. Ingestion
    print("\n[Data Ingestion] Loading raw data & building master entities...")
    loader = BerkaDataLoader()
    master = loader.build_master_loan_entities()
    trans = loader.load_transactions()
    print(f"Master entities: {master.shape} | Transactions: {trans.shape}")

    # 2. Cohort definition & truncation detection
    print("\n[Cohort Definition] Assigning cohorts and complete-window eligibility...")
    cohort_df = assign_cohorts_and_eligibility(master)
    summary = get_cohort_summary(cohort_df)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 3. Feature extraction
    print("\n[Feature Engineering] Extracting deterministic features across windows...")
    extractor = FeatureExtractor(cohort_df, trans)
    full_df = extractor.build_full_feature_dataset()
    print(f"Full feature dataset: {full_df.shape}")

    # 4. Data dictionary export
    dict_df = get_data_dictionary_df()
    dict_path = os.path.join(PROCESSED_DIR, 'data_dictionary.csv')
    dict_df.to_csv(dict_path, index=False)
    print(f"Saved data dictionary ({len(dict_df)} features) to {dict_path}")

    # Save full master feature dataset
    master_path = os.path.join(PROCESSED_DIR, 'features_master.csv')
    full_df.to_csv(master_path, index=False)
    print(f"Saved master features to {master_path}")

    # Save window-specific modeling cuts for Primary (Completed) Cohort
    comp_mask = full_df['is_completed_cohort']
    static_pre_cols = [c for c in full_df.columns if c.startswith('feat_') and not any(f'_w{w}_' in c for w in [30, 60, 90])]
    id_cols = ['loan_id', 'account_id', 'client_id', 'loan_date', 'status', 'target_completed']

    for w in [30, 60, 90]:
        w_cols = [c for c in full_df.columns if f'_w{w}_' in c]
        export_cols = id_cols + static_pre_cols + w_cols
        w_df = full_df[comp_mask & full_df[f'eligible_{w}d']][export_cols].copy()
        out_path = os.path.join(PROCESSED_DIR, f'completed_cohort_w{w}d.csv')
        w_df.to_csv(out_path, index=False)
        print(f"Saved Primary Completed Cohort (Window {w}d): {w_df.shape} to {out_path}")

    # Save Secondary Sensitivity Expanded Cohort (Window 90d, N=656)
    exp_mask = full_df['is_expanded_cohort_90d']
    exp_id_cols = ['loan_id', 'account_id', 'client_id', 'loan_date', 'status', 'target_expanded']
    w90_cols = [c for c in full_df.columns if '_w90_' in c]
    exp_cols = exp_id_cols + static_pre_cols + w90_cols
    exp_df = full_df[exp_mask][exp_cols].copy()
    exp_out_path = os.path.join(PROCESSED_DIR, 'expanded_cohort_w90d.csv')
    exp_df.to_csv(exp_out_path, index=False)
    print(f"Saved Secondary Expanded Cohort (Window 90d): {exp_df.shape} to {exp_out_path}")

    print("\nPipeline execution complete.")


if __name__ == '__main__':
    main()
