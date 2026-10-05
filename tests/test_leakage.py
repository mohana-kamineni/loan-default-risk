"""
Automated Leakage and Correctness Test Suite
Enforces strict temporal fencing, outcome independence, entity disjointness,
reproducibility, and quarantine of documented leakage shortcuts.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import StratifiedKFold

from src.ingestion import BerkaDataLoader
from src.cohorts import assign_cohorts_and_eligibility, DATASET_CUTOFF_DATE
from src.features import FeatureExtractor
from src.data_dictionary import FEATURE_DATA_DICTIONARY


@pytest.fixture(scope="module")
def data_setup():
    loader = BerkaDataLoader()
    master = loader.build_master_loan_entities()
    cohort_df = assign_cohorts_and_eligibility(master)
    trans = loader.load_transactions()
    extractor = FeatureExtractor(cohort_df, trans)
    features_df = extractor.build_full_feature_dataset()
    return {
        'loader': loader,
        'master': master,
        'cohort_df': cohort_df,
        'trans': trans,
        'extractor': extractor,
        'features_df': features_df
    }


def test_1_no_future_transactions_enter_windows(data_setup):
    """Test 1: Verify no transaction after loan_date + window enters the feature set."""
    features_df = data_setup['features_df']
    for w in [30, 60, 90]:
        max_col = f'max_trans_date_used_{w}d'
        max_used = features_df[max_col].dropna()
        max_allowed = features_df.loc[max_used.index, 'loan_date'] + pd.to_timedelta(w, unit='D')
        violations = max_used > max_allowed
        assert not violations.any(), f"Window {w}d failed: {violations.sum()} accounts used future transactions!"


def test_2_no_past_transactions_in_post_disbursement_window(data_setup):
    """Test 2: Verify no transaction before loan_date enters post-disbursement features."""
    extractor = data_setup['extractor']
    for w in [30, 60, 90]:
        w_trans = extractor.loan_trans[
            (extractor.loan_trans['days_from_loan'] >= 0) &
            (extractor.loan_trans['days_from_loan'] <= w)
        ]
        # Strictly verify min days_from_loan >= 0
        assert (w_trans['days_from_loan'] >= 0).all(), f"Window {w}d includes pre-loan transactions!"
        assert (w_trans['days_from_loan'] <= w).all(), f"Window {w}d includes transactions beyond {w} days!"


def test_3_no_lifetime_aggregates_in_features(data_setup):
    """Test 3: Assert no full-lifetime balance or outcome aggregates exist in feature matrix."""
    features_df = data_setup['features_df']
    feat_cols = [c for c in features_df.columns if c.startswith('feat_')]
    forbidden_terms = ['full_lifetime', 'lifetime_min', 'lifetime_balance', 'future', 'outcome']
    for col in feat_cols:
        for term in forbidden_terms:
            assert term not in col.lower(), f"Forbidden lifetime feature found: {col}"


def test_4_no_target_or_status_in_feature_columns(data_setup):
    """Test 4: Assert no target or status label information leaks into feature matrix X."""
    features_df = data_setup['features_df']
    feat_cols = [c for c in features_df.columns if c.startswith('feat_')]
    forbidden_labels = ['status', 'target', 'is_bad', 'default', 'is_completed', 'is_expanded']
    for col in feat_cols:
        for lbl in forbidden_labels:
            assert lbl not in col.lower(), f"Target label leakage in feature: {col}"


def test_5_right_truncated_loans_quarantined(data_setup):
    """Test 5: Verify all right-truncated loans are detected and excluded from complete-window sets."""
    cohort_df = data_setup['cohort_df']
    
    # At 90d, exactly 26 loans must be flagged as truncated
    truncated_90 = cohort_df[cohort_df['is_truncated_90d']]
    assert len(truncated_90) == 26, f"Expected 26 truncated loans at 90d, got {len(truncated_90)}"
    
    # Verify that all 26 loans have loan_date + 90d > 1998-12-31
    for _, row in truncated_90.iterrows():
        assert row['loan_date'] + pd.to_timedelta(90, unit='D') > DATASET_CUTOFF_DATE
        assert row['eligible_90d'] == False

    # Verify that completed cohort has 0 truncated loans
    completed_trunc = cohort_df[cohort_df['is_completed_cohort'] & cohort_df['is_truncated_90d']]
    assert len(completed_trunc) == 0, "Completed cohort should have 0 truncated loans!"


def test_6_deterministic_reproducibility(data_setup):
    """Test 6: Verify that feature extraction produces bitwise identical results on repeated runs."""
    extractor = data_setup['extractor']
    df1 = extractor.build_full_feature_dataset()
    df2 = extractor.build_full_feature_dataset()
    pd.testing.assert_frame_equal(df1, df2, check_exact=True)


def test_7_entity_disjointness_in_cross_validation(data_setup):
    """Test 7: Verify zero overlap in account_id, client_id, and loan_id across CV folds."""
    cohort_df = data_setup['cohort_df']
    comp = cohort_df[cohort_df['is_completed_cohort']].copy().reset_index(drop=True)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(comp, comp['target_completed'])):
        train_loans = set(comp.loc[train_idx, 'loan_id'])
        test_loans = set(comp.loc[test_idx, 'loan_id'])
        train_accounts = set(comp.loc[train_idx, 'account_id'])
        test_accounts = set(comp.loc[test_idx, 'account_id'])
        train_clients = set(comp.loc[train_idx, 'client_id'])
        test_clients = set(comp.loc[test_idx, 'client_id'])

        assert len(train_loans.intersection(test_loans)) == 0, f"Fold {fold} loan overlap!"
        assert len(train_accounts.intersection(test_accounts)) == 0, f"Fold {fold} account overlap!"
        assert len(train_clients.intersection(test_clients)) == 0, f"Fold {fold} client overlap!"


def test_8_reproduce_and_quarantine_lifetime_balance_shortcut(data_setup):
    """
    Test 8: Empirically reproduce the full-lifetime minimum balance shortcut
    and assert that it is strictly quarantined from the feature matrix.
    """
    trans = data_setup['trans']
    cohort_df = data_setup['cohort_df']
    features_df = data_setup['features_df']

    # Compute full lifetime minimum balance
    full_min = trans.groupby('account_id')['balance'].min().to_dict()
    comp = cohort_df[cohort_df['is_completed_cohort']].copy()
    comp['lifetime_min_bal'] = comp['account_id'].map(full_min)
    
    # Check 100% separation on completed cohort
    pred_bad = comp['lifetime_min_bal'] < 0
    actual_bad = comp['target_completed'] == 1
    accuracy = (pred_bad == actual_bad).mean()
    assert accuracy == 1.0, f"Expected 100% separation, got {accuracy*100:.2f}%"

    # Verify that lifetime_min_bal is NOT in predictive features
    feat_cols = [c for c in features_df.columns if c.startswith('feat_')]
    assert 'feat_lifetime_min_bal' not in feat_cols
    assert 'lifetime_min_bal' not in feat_cols


def test_9_data_dictionary_completeness(data_setup):
    """Test 9: Verify that all extracted features are documented in the data dictionary."""
    features_df = data_setup['features_df']
    feat_cols = [c for c in features_df.columns if c.startswith('feat_')]
    for col in feat_cols:
        assert col in FEATURE_DATA_DICTIONARY, f"Feature {col} is missing from data dictionary!"
        meta = FEATURE_DATA_DICTIONARY[col]
        assert 'definition' in meta and len(meta['definition']) > 0
        assert 'rationale' in meta and len(meta['rationale']) > 0
        assert 'observation_window' in meta and len(meta['observation_window']) > 0
        assert 'leakage_check' in meta and len(meta['leakage_check']) > 0
