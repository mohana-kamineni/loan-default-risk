"""
Feature Metadata & Data Dictionary
Defines every feature with:
1. Definition
2. Rationale / Predictive Information
3. Observation Window
4. Leakage Verification Check
"""

import pandas as pd

FEATURE_DATA_DICTIONARY = {
    # --- STATIC ORIGINATION FEATURES (Known at or before loan disbursement) ---
    'feat_loan_amount': {
        'category': 'Static / Origination',
        'definition': 'Total principal amount of the loan granted in CZK.',
        'rationale': 'Higher principal loans pose greater repayment burdens and loss severity.',
        'observation_window': 'At origination (t = 0)',
        'leakage_check': 'Contract variable fixed at origination date; independent of repayment.'
    },
    'feat_duration_months': {
        'category': 'Static / Origination',
        'definition': 'Scheduled loan repayment term in months (12, 24, 36, 48, 60).',
        'rationale': 'Longer loan terms increase cumulative default probability and exposure time.',
        'observation_window': 'At origination (t = 0)',
        'leakage_check': 'Fixed contractual duration at grant; does not change based on outcome.'
    },
    'feat_monthly_payment': {
        'category': 'Static / Origination',
        'definition': 'Contractual monthly installment payment in CZK.',
        'rationale': 'Represents the regular monthly liquidity drain required to service debt.',
        'observation_window': 'At origination (t = 0)',
        'leakage_check': 'Contractual payment fixed at grant.'
    },
    'feat_payment_to_amount_ratio': {
        'category': 'Static / Origination',
        'definition': 'Ratio of monthly payment to total loan amount (payments / amount).',
        'rationale': 'Proxy for monthly amortization velocity and interest burden.',
        'observation_window': 'At origination (t = 0)',
        'leakage_check': 'Ratio of two fixed origination contract variables.'
    },
    'feat_account_age_days': {
        'category': 'Static / Origination',
        'definition': 'Days elapsed between account opening date and loan grant date.',
        'rationale': 'Tenure proxy: seasoned banking relationships often exhibit lower default risk.',
        'observation_window': 'Pre-origination (t <= 0)',
        'leakage_check': 'Difference between loan_date and account_date; both established at or before t=0.'
    },
    'feat_client_age_years': {
        'category': 'Static / Origination',
        'definition': 'Age of the primary borrower (OWNER client) in years at loan grant.',
        'rationale': 'Demographic maturity and life-stage financial stability indicator.',
        'observation_window': 'At origination (t = 0)',
        'leakage_check': 'Derived from client birth_number and loan origination date.'
    },
    'feat_client_is_female': {
        'category': 'Static / Origination',
        'definition': 'Binary indicator (1.0 = Female, 0.0 = Male) derived from birth number.',
        'rationale': 'Standard demographic control in credit risk research.',
        'observation_window': 'At origination (t = 0)',
        'leakage_check': 'Derived from client birth number.'
    },
    'feat_district_salary': {
        'category': 'Static / Origination',
        'definition': 'Average monthly salary (in CZK) in the bank branch district (A11).',
        'rationale': 'Macroeconomic affluence indicator of borrower location.',
        'observation_window': 'Regional baseline at origination',
        'leakage_check': 'District-level census demographic table (district.asc).'
    },
    'feat_district_unemployment': {
        'category': 'Static / Origination',
        'definition': 'Unemployment rate in branch district (A12 for <=1995, A13 for >=1996).',
        'rationale': 'Regional labor market distress directly impacts borrower repayment capacity.',
        'observation_window': 'Regional baseline aligned with loan year',
        'leakage_check': 'Year-matched regional demographic metric; no post-1996 data used for early loans.'
    },
    'feat_district_crimes': {
        'category': 'Static / Origination',
        'definition': 'Number of committed crimes in district (A15 for <=1995, A16 for >=1996).',
        'rationale': 'Regional socioeconomic stability proxy.',
        'observation_window': 'Regional baseline aligned with loan year',
        'leakage_check': 'Year-matched regional demographic metric.'
    },
    'feat_district_entrepreneurs': {
        'category': 'Static / Origination',
        'definition': 'Number of entrepreneurs per 1000 inhabitants in branch district (A14).',
        'rationale': 'Local business density and commercial economic vitality indicator.',
        'observation_window': 'Regional baseline at origination',
        'leakage_check': 'District-level census metric.'
    },

    # --- PRE-LOAN CONTROLS (Strictly t < 0) ---
    'feat_had_pre_loan_overdraft': {
        'category': 'Pre-Loan Control (Ablation)',
        'definition': 'Binary flag: 1.0 if account had balance < 0 strictly before loan_date, 0.0 otherwise.',
        'rationale': 'Core ablation variable: tests whether post-disbursement signals add value beyond prior distress.',
        'observation_window': 'Pre-loan history (days_from_loan < 0)',
        'leakage_check': 'Strictly filters trans_date < loan_date. Zero post-loan records accessed.'
    },
    'feat_pre_loan_tx_count': {
        'category': 'Pre-Loan Control',
        'definition': 'Total count of transactions recorded on the account prior to loan_date.',
        'rationale': 'Measures pre-existing account activity and relationship depth.',
        'observation_window': 'Pre-loan history (days_from_loan < 0)',
        'leakage_check': 'Strictly filters trans_date < loan_date.'
    },

    # --- POST-DISBURSEMENT DYNAMIC BEHAVIOURAL FEATURES (t in [0, W] days) ---
    # Window W in {30, 60, 90} days
}

def generate_window_feature_dictionary(w: int) -> dict:
    return {
        f'feat_w{w}_trans_count': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Total number of transactions occurring in [0, {w}] days post-disbursement.',
            'rationale': 'Drop in transaction frequency signals account dormancy, disuse, or salary cessation.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Strictly filtered: 0 <= days_from_loan <= {w}. No future transaction accessed.'
        },
        f'feat_w{w}_inflow_total': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Total sum of credit transactions (type == PRIJEM) in [0, {w}] days.',
            'rationale': 'Direct measure of incoming income, salary deposits, or business receipts.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Strictly filtered: 0 <= days_from_loan <= {w}.'
        },
        f'feat_w{w}_outflow_total': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Total sum of debit transactions (type in [VYDAJ, VYBER]) in [0, {w}] days.',
            'rationale': 'Total living expenses, withdrawals, and debt obligations leaving the account.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Strictly filtered: 0 <= days_from_loan <= {w}.'
        },
        f'feat_w{w}_net_cash_flow': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Net cash flow (inflow_total - outflow_total) in [0, {w}] days.',
            'rationale': 'Persistent negative cash flow indicates structural liquidity depletion.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Strictly filtered: 0 <= days_from_loan <= {w}.'
        },
        f'feat_w{w}_outflow_to_inflow_ratio': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Ratio of outflows to inflows (outflow / (inflow + 1.0)) in [0, {w}] days.',
            'rationale': 'Cash burn multiplier: values > 1.0 indicate account is being drained faster than replenished.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Strictly filtered: 0 <= days_from_loan <= {w}.'
        },
        f'feat_w{w}_cash_withdrawal_amount': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Total CZK withdrawn in cash (cash counter VYBER or ATM VYBER KARTOU) in [0, {w}] days.',
            'rationale': 'Rapid cash extraction is a recognized behavioral red flag for distress.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Strictly filtered: 0 <= days_from_loan <= {w}.'
        },
        f'feat_w{w}_cash_withdrawal_ratio': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Proportion of total debits executed in cash (cash_w / (outflows + 1.0)) in [0, {w}] days.',
            'rationale': 'Distinguishes structured automated debits from discretionary/panicked cash runs.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Strictly filtered: 0 <= days_from_loan <= {w}.'
        },
        f'feat_w{w}_balance_mean': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Mean recorded transaction balance in [0, {w}] days.',
            'rationale': 'General liquidity level maintained across normal day-to-day operations.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Strictly filtered: 0 <= days_from_loan <= {w}.'
        },
        f'feat_w{w}_balance_min': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Minimum recorded balance within [0, {w}] days.',
            'rationale': 'Measures proximity to zero/overdraft threshold within the observation window.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Window minimum only. NEVER uses full-lifetime balance.'
        },
        f'feat_w{w}_balance_slope': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Linear daily slope of balance trajectory over [0, {w}] days.',
            'rationale': 'Captures trajectory momentum: negative slope reflects declining reserves.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Strictly filtered: 0 <= days_from_loan <= {w}.'
        },
        f'feat_w{w}_had_negative_balance': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Binary flag: 1.0 if balance_min < 0 in [0, {w}] days, 0.0 otherwise.',
            'rationale': 'Distress marker: indicates account entered unauthorized overdraft within window.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Window-bounded indicator only; not a full-lifetime aggregate.'
        },
        f'feat_w{w}_had_sanction_interest': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Binary flag: 1.0 if SANKC. UROK transaction occurred in [0, {w}] days, 0.0 otherwise.',
            'rationale': 'Administrative fee marker for overdraft or irregular account servicing.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Strictly filtered: 0 <= days_from_loan <= {w}.'
        },
        f'feat_w{w}_min_bal_to_payment_ratio': {
            'category': f'Dynamic Behavioural (Window {w}d)',
            'definition': f'Ratio of balance_min to monthly loan payment (balance_min / payments).',
            'rationale': 'Liquidity runway: how many monthly loan payments can be covered by the lowest balance.',
            'observation_window': f'Day 0 to Day {w} post-disbursement',
            'leakage_check': f'Combines window min balance with contractual payment.'
        }
    }

# Build full dictionary
for _w in [30, 60, 90]:
    FEATURE_DATA_DICTIONARY.update(generate_window_feature_dictionary(_w))


def get_data_dictionary_df() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(FEATURE_DATA_DICTIONARY, orient='index')
    df.index.name = 'feature_name'
    return df.reset_index()


if __name__ == '__main__':
    dict_df = get_data_dictionary_df()
    print(f"Data dictionary generated: {len(dict_df)} features documented.")
    print(dict_df[['feature_name', 'category', 'observation_window']].head(15))
