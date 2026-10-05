"""
Deterministic Feature Extraction Pipeline
Extracts static origination features, pre-loan controls, and windowed post-disbursement
behavioural features for W in {30, 60, 90} days.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Tuple


def compute_balance_slope(days: np.ndarray, balances: np.ndarray) -> float:
    """Compute linear slope of balance trajectory over elapsed days."""
    if len(days) < 2:
        return 0.0
    # Center days
    x = days - np.mean(days)
    denom = np.sum(x ** 2)
    if denom == 0:
        return 0.0
    y = balances - np.mean(balances)
    return float(np.sum(x * y) / denom)


class FeatureExtractor:
    def __init__(self, cohort_df: pd.DataFrame, trans_df: pd.DataFrame):
        """
        cohort_df: output of assign_cohorts_and_eligibility (682 rows)
        trans_df: raw transactions dataframe (1,056,320 rows)
        """
        self.cohort = cohort_df.copy()
        self.trans = trans_df.copy()
        # Pre-filter transactions belonging to the 682 loan accounts
        loan_accounts = set(self.cohort['account_id'])
        self.loan_trans = self.trans[self.trans['account_id'].isin(loan_accounts)].copy()
        # Merge loan date onto transactions to compute exact relative days
        loan_date_map = self.cohort.set_index('account_id')['loan_date'].to_dict()
        self.loan_trans['loan_date'] = self.loan_trans['account_id'].map(loan_date_map)
        self.loan_trans['days_from_loan'] = (self.loan_trans['trans_date'] - self.loan_trans['loan_date']).dt.days

    def extract_static_and_pre_loan_features(self) -> pd.DataFrame:
        """
        Extract static contract, borrower demographic, and pre-loan account features.
        These features are strictly determined at or before loan_date (t <= 0).
        """
        df = self.cohort.copy()

        # Contract variables
        df['feat_loan_amount'] = df['amount'].astype(float)
        df['feat_duration_months'] = df['duration_months'].astype(float)
        df['feat_monthly_payment'] = df['payments'].astype(float)
        df['feat_payment_to_amount_ratio'] = df['payments'] / df['amount']

        # Account maturity & Borrower demographics
        df['feat_account_age_days'] = df['account_age_days_at_loan'].astype(float)
        df['feat_client_age_years'] = df['client_age_years_at_loan'].astype(float)
        df['feat_client_is_female'] = (df['gender'] == 'F').astype(float)

        # Regional macro indicators (branch district)
        # Select unemployment based on loan year (1993-1995 -> unemploy_95; 1996-1998 -> unemploy_96)
        loan_year = df['loan_date'].dt.year
        df['feat_district_salary'] = df['avg_salary'].astype(float)
        df['feat_district_unemployment'] = np.where(
            loan_year >= 1996, df['unemploy_96'], df['unemploy_95']
        ).astype(float)
        df['feat_district_crimes'] = np.where(
            loan_year >= 1996, df['crimes_96'], df['crimes_95']
        ).astype(float)
        df['feat_district_entrepreneurs'] = df['entrepreneurs_per_1000'].astype(float)

        # Pre-loan transaction behaviour (strictly t < 0)
        pre_trans = self.loan_trans[self.loan_trans['days_from_loan'] < 0]
        
        pre_grouped = pre_trans.groupby('account_id').agg(
            pre_min_bal=('balance', 'min'),
            pre_tx_count=('trans_id', 'count')
        )

        df['feat_had_pre_loan_overdraft'] = df['account_id'].map(
            lambda aid: 1.0 if aid in pre_grouped.index and pre_grouped.loc[aid, 'pre_min_bal'] < 0 else 0.0
        )
        df['feat_pre_loan_tx_count'] = df['account_id'].map(
            lambda aid: float(pre_grouped.loc[aid, 'pre_tx_count']) if aid in pre_grouped.index else 0.0
        )

        return df

    def extract_window_features(self, window_days: int) -> pd.DataFrame:
        """
        Extract post-disbursement dynamic behavioural features for window W.
        Temporal boundary: 0 <= days_from_loan <= window_days.
        Day 0 is included as disbursement-day liquidity behavior.
        """
        # Strictly slice transactions within [0, window_days]
        w_trans = self.loan_trans[
            (self.loan_trans['days_from_loan'] >= 0) &
            (self.loan_trans['days_from_loan'] <= window_days)
        ].copy()

        # Sort for sequential calculations (slopes, trajectories)
        w_trans = w_trans.sort_values(['account_id', 'trans_date', 'trans_id'])

        records = {}
        grouped = w_trans.groupby('account_id')

        for aid, g in grouped:
            inflows = g[g['type'] == 'PRIJEM']['amount'].sum()
            outflows = g[g['type'].isin(['VYDAJ', 'VYBER'])]['amount'].sum()
            net_flow = inflows - outflows
            
            # Cash withdrawals
            cash_w = g[g['operation'].isin(['VYBER', 'VYBER KARTOU'])]['amount'].sum()
            
            # Sanction interest
            had_sankc = float((g['k_symbol'] == 'SANKC. UROK').any())

            # Balance metrics
            balances = g['balance'].values
            days = g['days_from_loan'].values
            min_bal = float(np.min(balances)) if len(balances) > 0 else 0.0
            mean_bal = float(np.mean(balances)) if len(balances) > 0 else 0.0
            slope = compute_balance_slope(days, balances)

            # Max transaction date used (for leakage test validation)
            max_date_used = g['trans_date'].max()

            records[aid] = {
                f'feat_w{window_days}_trans_count': float(len(g)),
                f'feat_w{window_days}_inflow_total': float(inflows),
                f'feat_w{window_days}_outflow_total': float(outflows),
                f'feat_w{window_days}_net_cash_flow': float(net_flow),
                f'feat_w{window_days}_outflow_to_inflow_ratio': float(outflows / (inflows + 1.0)),
                f'feat_w{window_days}_cash_withdrawal_amount': float(cash_w),
                f'feat_w{window_days}_cash_withdrawal_ratio': float(cash_w / (outflows + 1.0)),
                f'feat_w{window_days}_balance_mean': float(mean_bal),
                f'feat_w{window_days}_balance_min': float(min_bal),
                f'feat_w{window_days}_balance_slope': float(slope),
                f'feat_w{window_days}_had_negative_balance': float(min_bal < 0),
                f'feat_w{window_days}_had_sanction_interest': float(had_sankc),
                f'max_trans_date_used_{window_days}d': max_date_used
            }

        w_df = pd.DataFrame.from_dict(records, orient='index')
        w_df.index.name = 'account_id'
        w_df = w_df.reset_index()

        # Join with cohort to guarantee all 682 accounts are present
        res = self.cohort[['account_id', 'loan_id', 'loan_date', 'payments']].merge(
            w_df, on='account_id', how='left'
        )

        # For any account with 0 transactions in window (none in reality, but for safety fill defaults)
        res[f'feat_w{window_days}_trans_count'] = res[f'feat_w{window_days}_trans_count'].fillna(0.0)
        res[f'feat_w{window_days}_inflow_total'] = res[f'feat_w{window_days}_inflow_total'].fillna(0.0)
        res[f'feat_w{window_days}_outflow_total'] = res[f'feat_w{window_days}_outflow_total'].fillna(0.0)
        res[f'feat_w{window_days}_net_cash_flow'] = res[f'feat_w{window_days}_net_cash_flow'].fillna(0.0)
        res[f'feat_w{window_days}_outflow_to_inflow_ratio'] = res[f'feat_w{window_days}_outflow_to_inflow_ratio'].fillna(0.0)
        res[f'feat_w{window_days}_cash_withdrawal_amount'] = res[f'feat_w{window_days}_cash_withdrawal_amount'].fillna(0.0)
        res[f'feat_w{window_days}_cash_withdrawal_ratio'] = res[f'feat_w{window_days}_cash_withdrawal_ratio'].fillna(0.0)
        res[f'feat_w{window_days}_balance_mean'] = res[f'feat_w{window_days}_balance_mean'].fillna(0.0)
        res[f'feat_w{window_days}_balance_min'] = res[f'feat_w{window_days}_balance_min'].fillna(0.0)
        res[f'feat_w{window_days}_balance_slope'] = res[f'feat_w{window_days}_balance_slope'].fillna(0.0)
        res[f'feat_w{window_days}_had_negative_balance'] = res[f'feat_w{window_days}_had_negative_balance'].fillna(0.0)
        res[f'feat_w{window_days}_had_sanction_interest'] = res[f'feat_w{window_days}_had_sanction_interest'].fillna(0.0)

        # Interaction feature: minimum balance to monthly payment ratio (buffer ratio)
        res[f'feat_w{window_days}_min_bal_to_payment_ratio'] = (
            res[f'feat_w{window_days}_balance_min'] / res['payments']
        ).astype(float)

        return res

    def build_full_feature_dataset(self) -> pd.DataFrame:
        """Combine static, pre-loan, and all window features into a master dataset."""
        df = self.extract_static_and_pre_loan_features()
        for w in [30, 60, 90]:
            w_df = self.extract_window_features(window_days=w)
            # Drop redundant loan columns from w_df before merge
            drop_cols = ['loan_id', 'loan_date', 'payments']
            w_df_to_merge = w_df.drop(columns=[c for c in drop_cols if c in w_df.columns])
            df = df.merge(w_df_to_merge, on='account_id', how='left')
        return df


if __name__ == '__main__':
    from src.ingestion import BerkaDataLoader
    from src.cohorts import assign_cohorts_and_eligibility

    loader = BerkaDataLoader()
    master = loader.build_master_loan_entities()
    cohort_df = assign_cohorts_and_eligibility(master)
    trans = loader.load_transactions()

    extractor = FeatureExtractor(cohort_df, trans)
    full_features = extractor.build_full_feature_dataset()
    print(f"Full feature dataset built successfully: {full_features.shape}")
    feat_cols = [c for c in full_features.columns if c.startswith('feat_')]
    print(f"Total features created: {len(feat_cols)}")
    print("Features list:\n", feat_cols)
