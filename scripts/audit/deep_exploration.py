import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'berka-dataset')

def parse_date(val):
    s = str(val).strip()
    return pd.Timestamp(year=1900 + int(s[0:2]), month=int(s[2:4]), day=int(s[4:6]))

def run_deep_exploration():
    loan = pd.read_csv(os.path.join(DATA_DIR, 'loan.asc'), sep=';')
    account = pd.read_csv(os.path.join(DATA_DIR, 'account.asc'), sep=';')
    disp = pd.read_csv(os.path.join(DATA_DIR, 'disp.asc'), sep=';')
    client = pd.read_csv(os.path.join(DATA_DIR, 'client.asc'), sep=';')
    card = pd.read_csv(os.path.join(DATA_DIR, 'card.asc'), sep=';')
    district = pd.read_csv(os.path.join(DATA_DIR, 'district.asc'), sep=';')
    order = pd.read_csv(os.path.join(DATA_DIR, 'order.asc'), sep=';')
    trans = pd.read_csv(os.path.join(DATA_DIR, 'trans.asc'), sep=';', low_memory=False)

    loan['loan_date'] = loan['date'].apply(parse_date)
    account['account_date'] = account['date'].apply(parse_date)
    trans['trans_date'] = trans['date'].apply(parse_date)

    print("=== DEEP EXPLORATION FOR DESIGN DECISIONS ===\n")

    # 1. Null check across all tables
    print("--- 1. Null Check Across All Tables ---")
    tables = {
        'loan': loan, 'account': account, 'disp': disp, 
        'client': client, 'card': card, 'district': district, 
        'order': order, 'trans': trans
    }
    for name, df in tables.items():
        null_counts = df.isnull().sum()
        cols_with_null = null_counts[null_counts > 0]
        if len(cols_with_null) > 0:
            print(f"Table '{name}' nulls:")
            for col, n in cols_with_null.items():
                print(f"  {col}: {n} ({n/len(df)*100:.2f}%)")
        else:
            print(f"Table '{name}': No nulls")

    # Check district table question marks (PKDD'99 district table often has '?' for missing in A12, A15)
    print("\nDistrict table object columns / string check for '?':")
    for col in district.columns:
        q_count = (district[col].astype(str) == '?').sum()
        if q_count > 0:
            print(f"  District col {col} has {q_count} '?' values")

    # 2. Timing of Default / First Negative Balance
    print("\n--- 2. Timing of First Negative Balance in Bad Loans (B and D) ---")
    bad_loans = loan[loan['status'].isin(['B', 'D'])].copy()
    bad_loan_trans = bad_loans[['loan_id', 'account_id', 'loan_date', 'status', 'duration']].merge(
        trans[['account_id', 'trans_date', 'balance', 'type', 'k_symbol']],
        on='account_id',
        how='inner'
    )
    bad_loan_trans['days_since_loan'] = (bad_loan_trans['trans_date'] - bad_loan_trans['loan_date']).dt.days

    # Filter to transactions where balance < 0
    neg_trans = bad_loan_trans[bad_loan_trans['balance'] < 0]
    first_neg = neg_trans.groupby('loan_id').agg(
        first_neg_date=('trans_date', 'min'),
        first_neg_days=('days_since_loan', 'min'),
        min_balance=('balance', 'min')
    ).reset_index()
    first_neg = bad_loans.merge(first_neg, on='loan_id', how='left')

    print("First negative balance days since loan disbursement for Bad Loans (B and D):")
    for st in ['B', 'D']:
        sub = first_neg[first_neg['status'] == st]
        print(f"\nStatus {st} (n={len(sub)}):")
        print(f"  Min days to negative:    {sub['first_neg_days'].min()}")
        print(f"  25% percentile:          {sub['first_neg_days'].quantile(0.25):.1f} days")
        print(f"  Median days to negative: {sub['first_neg_days'].median():.1f} days")
        print(f"  75% percentile:          {sub['first_neg_days'].quantile(0.75):.1f} days")
        print(f"  Max days to negative:    {sub['first_neg_days'].max()} days")
        
        # How many go negative within 30, 60, 90, 180, 365 days?
        print(f"  Negative <= 30 days:  {(sub['first_neg_days'] <= 30).sum()} / {len(sub)} ({(sub['first_neg_days'] <= 30).mean()*100:.1f}%)")
        print(f"  Negative <= 60 days:  {(sub['first_neg_days'] <= 60).sum()} / {len(sub)} ({(sub['first_neg_days'] <= 60).mean()*100:.1f}%)")
        print(f"  Negative <= 90 days:  {(sub['first_neg_days'] <= 90).sum()} / {len(sub)} ({(sub['first_neg_days'] <= 90).mean()*100:.1f}%)")
        print(f"  Negative <= 180 days: {(sub['first_neg_days'] <= 180).sum()} / {len(sub)} ({(sub['first_neg_days'] <= 180).mean()*100:.1f}%)")
        print(f"  Negative <= 365 days: {(sub['first_neg_days'] <= 365).sum()} / {len(sub)} ({(sub['first_neg_days'] <= 365).mean()*100:.1f}%)")
        print(f"  Negative > 365 days:  {(sub['first_neg_days'] > 365).sum()} / {len(sub)} ({(sub['first_neg_days'] > 365).mean()*100:.1f}%)")

    # Did any Good loans (A or C) EVER have a negative balance in the early window?
    good_loans = loan[loan['status'].isin(['A', 'C'])].copy()
    good_loan_trans = good_loans[['loan_id', 'account_id', 'loan_date', 'status']].merge(
        trans[['account_id', 'trans_date', 'balance']],
        on='account_id',
        how='inner'
    )
    good_loan_trans['days_since_loan'] = (good_loan_trans['trans_date'] - good_loan_trans['loan_date']).dt.days
    
    print("\n--- 3. Early Window Negative Balances Across ALL Loans ---")
    for w in [30, 60, 90]:
        early_trans = loan[['loan_id', 'status', 'account_id', 'loan_date']].merge(
            trans[['account_id', 'trans_date', 'balance']], on='account_id'
        )
        early_trans['days'] = (early_trans['trans_date'] - early_trans['loan_date']).dt.days
        early_w = early_trans[(early_trans['days'] >= 0) & (early_trans['days'] <= w)]
        min_in_w = early_w.groupby(['loan_id', 'status'])['balance'].min().reset_index()
        neg_in_w = min_in_w[min_in_w['balance'] < 0]
        print(f"In window [0, {w}] days, number of loans with min balance < 0:")
        print(neg_in_w['status'].value_counts())

    # 4. Check Order table vs Loan Payments
    print("\n--- 4. Permanent Order Table Deep Dive ---")
    uver_orders = order[order['k_symbol'] == 'UVER']
    loan_order_merge = loan.merge(uver_orders, on='account_id', how='left')
    print(f"Loans with exactly 1 UVER order: {(loan_order_merge.groupby('loan_id').size() == 1).sum()} / {len(loan)}")
    # Compare order amount with loan payments
    loan_order_merge['diff'] = (loan_order_merge['amount_y'] - loan_order_merge['payments']).abs()
    print(f"Max difference between order.amount and loan.payments: {loan_order_merge['diff'].max()}")
    print("Does order table have ANY timestamp or date column?")
    print(f"Order columns: {list(order.columns)}")
    print("Conclusion on order.asc: It is a static table without timestamps.")

    # 5. Transaction Types and k_symbols in Early Windows
    print("\n--- 5. Transaction Features Available in 30/60/90 Days ---")
    # For all loans, look at transaction diversity in [1, 90]
    early_90 = loan[['loan_id', 'account_id', 'loan_date']].merge(
        trans, on='account_id'
    )
    early_90['days'] = (early_90['trans_date'] - early_90['loan_date']).dt.days
    w90 = early_90[(early_90['days'] >= 0) & (early_90['days'] <= 90)]
    
    print("Transaction types in [0, 90] days:")
    print(w90['type'].value_counts(dropna=False))
    print("\nTransaction operations in [0, 90] days:")
    print(w90['operation'].value_counts(dropna=False))
    print("\nk_symbol in [0, 90] days:")
    print(w90['k_symbol'].value_counts(dropna=False))

    # Does 'SANKC. UROK' (sanction interest) appear in the early windows?
    for w in [30, 60, 90]:
        w_sub = early_90[(early_90['days'] >= 0) & (early_90['days'] <= w)]
        sankc = w_sub[w_sub['k_symbol'] == 'SANKC. UROK']
        print(f"Number of SANKC. UROK transactions in [0, {w}] days: {len(sankc)} across {sankc['loan_id'].nunique()} loans")
        if len(sankc) > 0:
            print(f"  Statuses: {loan[loan['loan_id'].isin(sankc['loan_id'])]['status'].value_counts().to_dict()}")

    # 6. Detailed comparison of completed vs running cohorts (Option A vs Option B)
    print("\n--- 6. Cohort Comparison: Completed (A,B) vs Running (C,D) ---")
    print("Year of loan disbursement by status:")
    loan['loan_year'] = loan['loan_date'].dt.year
    print(pd.crosstab(loan['loan_year'], loan['status'], margins=True))

    print("\nDuration distribution by status:")
    print(pd.crosstab(loan['duration'], loan['status'], margins=True))

    print("\nLoan amount summary by status:")
    print(loan.groupby('status')['amount'].describe())

    print("\nMonthly payments summary by status:")
    print(loan.groupby('status')['payments'].describe())

if __name__ == '__main__':
    run_deep_exploration()
