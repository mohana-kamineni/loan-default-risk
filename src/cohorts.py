"""
Loan Cohort Construction & Truncation-Aware Eligibility
Defines Primary (Completed N=234) and Secondary (Expanded N=656) cohorts
and flags observation eligibility across 30, 60, and 90-day windows.
"""

import pandas as pd
import numpy as np

DATASET_CUTOFF_DATE = pd.Timestamp('1998-12-31')


def assign_cohorts_and_eligibility(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the master loan entities dataframe (682 rows), assign:
    - target labels for completed (is_completed, default_completed)
    - target labels for expanded (is_expanded, default_expanded)
    - truncation flags and complete-window eligibility for W in {30, 60, 90} days.
    """
    df = master_df.copy()
    assert len(df) == 682, f"Expected 682 loans, got {len(df)}"

    # Target definitions
    # Primary cohort: Completed loans only (A vs B)
    df['is_completed_cohort'] = df['status'].isin(['A', 'B'])
    df['target_completed'] = np.where(
        df['status'] == 'B', 1,
        np.where(df['status'] == 'A', 0, np.nan)
    )

    # Expanded cohort: All loans (A+C vs B+D)
    df['target_expanded'] = np.where(
        df['status'].isin(['B', 'D']), 1, 0
    )

    # Days available until dataset cutoff
    df['days_until_dataset_cutoff'] = (DATASET_CUTOFF_DATE - df['loan_date']).dt.days

    # Complete-window eligibility for each window W in {30, 60, 90}
    for w in [30, 60, 90]:
        # A loan is right-truncated for window W if its observation window extends past 1998-12-31
        df[f'is_truncated_{w}d'] = df['days_until_dataset_cutoff'] < w
        # Eligible if not truncated
        df[f'eligible_{w}d'] = ~df[f'is_truncated_{w}d']

    # Secondary sensitivity cohort definition:
    # Defined at W=90 days where complete observation across all 3 windows is guaranteed:
    # 682 - 26 truncated = 656 loans
    df['is_expanded_cohort_90d'] = df['eligible_90d']

    return df


def get_cohort_summary(cohort_df: pd.DataFrame) -> dict:
    """Compute exact cohort and eligibility counts for reporting."""
    summary = {}
    
    # Completed cohort counts
    comp = cohort_df[cohort_df['is_completed_cohort']]
    summary['completed_total'] = len(comp)
    summary['completed_good_A'] = (comp['status'] == 'A').sum()
    summary['completed_bad_B'] = (comp['status'] == 'B').sum()
    summary['completed_default_rate'] = (comp['status'] == 'B').mean()

    # Truncation counts across all 682 loans
    for w in [30, 60, 90]:
        trunc = cohort_df[cohort_df[f'is_truncated_{w}d']]
        summary[f'truncated_{w}d_total'] = len(trunc)
        summary[f'truncated_{w}d_by_status'] = trunc['status'].value_counts().to_dict()
        summary[f'eligible_{w}d_total'] = cohort_df[f'eligible_{w}d'].sum()

    # Completed cohort eligibility (should be 234 for all windows)
    for w in [30, 60, 90]:
        comp_trunc = comp[comp[f'is_truncated_{w}d']]
        summary[f'completed_truncated_{w}d'] = len(comp_trunc)
        summary[f'completed_eligible_{w}d'] = len(comp) - len(comp_trunc)

    # Expanded cohort at 90d (N=656)
    exp_90 = cohort_df[cohort_df['is_expanded_cohort_90d']]
    summary['expanded_90d_total'] = len(exp_90)
    summary['expanded_90d_good'] = (exp_90['target_expanded'] == 0).sum()
    summary['expanded_90d_bad'] = (exp_90['target_expanded'] == 1).sum()
    summary['expanded_90d_default_rate'] = (exp_90['target_expanded'] == 1).mean()

    return summary


if __name__ == '__main__':
    from src.ingestion import BerkaDataLoader
    loader = BerkaDataLoader()
    master = loader.build_master_loan_entities()
    cohort_df = assign_cohorts_and_eligibility(master)
    summary = get_cohort_summary(cohort_df)
    print("=== COHORT & ELIGIBILITY AUDIT ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
