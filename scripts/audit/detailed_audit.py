import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), 'berka-dataset')

def parse_date(val):
    s = str(val).strip()
    if len(s) == 6:
        yy = int(s[0:2])
        mm = int(s[2:4])
        dd = int(s[4:6])
        year = 1900 + yy
        return pd.Timestamp(year=year, month=mm, day=dd)
    return pd.NaT

def run_audit():
    print("=================================================================")
    print("DETAILED AUDIT OF BERKA DATASET FOR EARLY DEFAULT PREDICTION")
    print("=================================================================\n")

    # Load tables
    print("--- 1. Loading raw tables ---")
    loan = pd.read_csv(os.path.join(DATA_DIR, 'loan.asc'), sep=';')
    account = pd.read_csv(os.path.join(DATA_DIR, 'account.asc'), sep=';')
    disp = pd.read_csv(os.path.join(DATA_DIR, 'disp.asc'), sep=';')
    client = pd.read_csv(os.path.join(DATA_DIR, 'client.asc'), sep=';')
    card = pd.read_csv(os.path.join(DATA_DIR, 'card.asc'), sep=';')
    district = pd.read_csv(os.path.join(DATA_DIR, 'district.asc'), sep=';')
    order = pd.read_csv(os.path.join(DATA_DIR, 'order.asc'), sep=';')
    trans = pd.read_csv(os.path.join(DATA_DIR, 'trans.asc'), sep=';', low_memory=False)

    print(f"loan:     {loan.shape}")
    print(f"account:  {account.shape}")
    print(f"disp:     {disp.shape}")
    print(f"client:   {client.shape}")
    print(f"card:     {card.shape}")
    print(f"district: {district.shape}")
    print(f"order:    {order.shape}")
    print(f"trans:    {trans.shape}\n")

    # Parse dates
    loan['loan_date'] = loan['date'].apply(parse_date)
    account['account_date'] = account['date'].apply(parse_date)
    trans['trans_date'] = trans['date'].apply(parse_date)

    print("--- 2. Loan Status Distribution & Target Analysis ---")
    status_counts = loan['status'].value_counts().sort_index()
    print("Status counts:")
    for st, cnt in status_counts.items():
        pct = cnt / len(loan) * 100
        print(f"  Status {st}: {cnt:4d} ({pct:6.2f}%)")
    
    print("\nGrouped Status (A+C vs B+D):")
    good_total = status_counts.get('A', 0) + status_counts.get('C', 0)
    bad_total = status_counts.get('B', 0) + status_counts.get('D', 0)
    print(f"  Good (A+C): {good_total:4d} ({good_total/len(loan)*100:6.2f}%)")
    print(f"  Bad  (B+D): {bad_total:4d} ({bad_total/len(loan)*100:6.2f}%)")
    print(f"  Total:      {len(loan)}")

    print("\nCompleted Loans Only (A vs B):")
    completed = loan[loan['status'].isin(['A', 'B'])].copy()
    comp_a = status_counts.get('A', 0)
    comp_b = status_counts.get('B', 0)
    comp_total = len(completed)
    print(f"  Status A (finished, OK):      {comp_a:4d} ({comp_a/comp_total*100:6.2f}%)")
    print(f"  Status B (finished, default): {comp_b:4d} ({comp_b/comp_total*100:6.2f}%)")
    print(f"  Total completed:              {comp_total:4d}")

    print("\nRunning Loans Only (C vs D):")
    running = loan[loan['status'].isin(['C', 'D'])].copy()
    run_c = status_counts.get('C', 0)
    run_d = status_counts.get('D', 0)
    run_total = len(running)
    print(f"  Status C (running, OK so far): {run_c:4d} ({run_c/run_total*100:6.2f}%)")
    print(f"  Status D (running, in debt):   {run_d:4d} ({run_d/run_total*100:6.2f}%)")
    print(f"  Total running:                 {run_total:4d}\n")

    print("--- 3. Loan Date and Duration Ranges ---")
    print(f"Loan dates range from {loan['loan_date'].min().date()} to {loan['loan_date'].max().date()}")
    print(f"Account creation dates range from {account['account_date'].min().date()} to {account['account_date'].max().date()}")
    print(f"Transaction dates range from {trans['trans_date'].min().date()} to {trans['trans_date'].max().date()}")

    for st in ['A', 'B', 'C', 'D']:
        sub = loan[loan['status'] == st]
        print(f"  Status {st}: n={len(sub):3d} | Loan Date: {sub['loan_date'].min().date()} to {sub['loan_date'].max().date()} | Durations: {sorted(sub['duration'].unique())}")

    # Check relational structure between loan, account, disp, client
    print("\n--- 4. Relational & Entity Disjointness Analysis ---")
    print(f"Total loans: {len(loan)}")
    print(f"Unique account_id in loan: {loan['account_id'].nunique()}")
    
    # Merge loan with disp
    loan_disp = loan.merge(disp, on='account_id', how='left')
    print(f"Loan-disp merged rows: {len(loan_disp)}")
    print(f"Disposition types for loan accounts:\n{loan_disp['type'].value_counts()}")
    
    owners = loan_disp[loan_disp['type'] == 'OWNER']
    users = loan_disp[loan_disp['type'] == 'DISPONENT']
    print(f"Unique loan accounts: {loan['account_id'].nunique()}")
    print(f"Unique owners for loan accounts: {owners['client_id'].nunique()}")
    print(f"Number of loan accounts with DISPONENT (secondary user): {users['account_id'].nunique()}")
    
    # Are there any clients associated with more than one loan?
    client_loan_counts = owners['client_id'].value_counts()
    print(f"Max loans per owner: {client_loan_counts.max()}")
    all_client_loan_counts = loan_disp['client_id'].value_counts()
    print(f"Max loan associations per client (owner or disponent): {all_client_loan_counts.max()}")

    # Check whether any client in disp has multiple accounts overall, and whether any of those involve loans
    client_acc_counts = disp.groupby('client_id')['account_id'].nunique()
    print(f"Clients with multiple accounts in entire database: {(client_acc_counts > 1).sum()} out of {len(client_acc_counts)}")
    multi_acc_clients = client_acc_counts[client_acc_counts > 1].index
    multi_acc_loan_disp = loan_disp[loan_disp['client_id'].isin(multi_acc_clients)]
    print(f"Clients with multiple accounts that appear in loan accounts: {multi_acc_loan_disp['client_id'].nunique()}")

    # Check Order table
    print("\n--- 5. Order Table Analysis ---")
    print(f"Order columns: {list(order.columns)}")
    print(f"Order head:\n{order.head(3)}")
    loan_orders = order[order['account_id'].isin(loan['account_id'])]
    print(f"Total order rows: {len(order)}")
    print(f"Orders for loan accounts: {len(loan_orders)}")
    print(f"Unique loan accounts with orders: {loan_orders['account_id'].nunique()} / {loan['account_id'].nunique()}")
    print(f"Order k_symbols in loan accounts:\n{loan_orders['k_symbol'].value_counts(dropna=False)}")

    # Check Transaction Coverage relative to Loan Date
    print("\n--- 6. Post-Disbursement Transaction Coverage ---")
    # For each loan, join with transactions of that account
    loan_trans = loan[['loan_id', 'account_id', 'loan_date', 'status', 'duration', 'amount', 'payments']].merge(
        trans[['trans_id', 'account_id', 'trans_date', 'type', 'operation', 'amount', 'balance', 'k_symbol']],
        on='account_id',
        how='left'
    )
    loan_trans['days_since_loan'] = (loan_trans['trans_date'] - loan_trans['loan_date']).dt.days

    # Check coverage for different windows:
    # Notice: What does "post-disbursement" mean?
    # Does day 0 (the loan date itself) count, or days 1 to W, or days 0 to W?
    # Let's check both [0, W] and [1, W]
    for w in [30, 60, 90]:
        # [0, w]
        in_w = loan_trans[(loan_trans['days_since_loan'] >= 0) & (loan_trans['days_since_loan'] <= w)]
        counts_per_loan = in_w.groupby('loan_id').size().reindex(loan['loan_id'], fill_value=0)
        print(f"Window [0, {w:2d}] days:")
        print(f"  Loans with 0 transactions:   {(counts_per_loan == 0).sum():3d}")
        print(f"  Loans with < 3 transactions: {(counts_per_loan < 3).sum():3d}")
        print(f"  Loans with < 5 transactions: {(counts_per_loan < 5).sum():3d}")
        print(f"  Min tx: {counts_per_loan.min():2d}, Median tx: {counts_per_loan.median():.1f}, Mean tx: {counts_per_loan.mean():.1f}, Max tx: {counts_per_loan.max():2d}")

        # Strictly post-disbursement [1, w]
        in_w_strict = loan_trans[(loan_trans['days_since_loan'] >= 1) & (loan_trans['days_since_loan'] <= w)]
        counts_per_loan_strict = in_w_strict.groupby('loan_id').size().reindex(loan['loan_id'], fill_value=0)
        print(f"Window [1, {w:2d}] days (strictly post-disbursement, excluding day 0):")
        print(f"  Loans with 0 transactions:   {(counts_per_loan_strict == 0).sum():3d}")
        print(f"  Loans with < 3 transactions: {(counts_per_loan_strict < 3).sum():3d}")
        print(f"  Min tx: {counts_per_loan_strict.min():2d}, Median tx: {counts_per_loan_strict.median():.1f}, Mean tx: {counts_per_loan_strict.mean():.1f}, Max tx: {counts_per_loan_strict.max():2d}")

    # Check Day 0 transactions (what happens on loan date?)
    day0 = loan_trans[loan_trans['days_since_loan'] == 0]
    print(f"\nTransactions occurring on loan date (day 0): {len(day0)}")
    print(f"Accounts with day 0 transactions: {day0['account_id'].nunique()} / {loan['account_id'].nunique()}")
    print("Day 0 transaction operations / k_symbols:")
    print(day0[['type', 'operation', 'k_symbol']].value_counts(dropna=False).head(10))

    # Pre-loan history
    pre_loan = loan_trans[loan_trans['days_since_loan'] < 0]
    pre_counts = pre_loan.groupby('loan_id').size().reindex(loan['loan_id'], fill_value=0)
    print(f"\nPre-loan transactions (< day 0):")
    print(f"  Loans with 0 pre-loan tx: {(pre_counts == 0).sum()}")
    print(f"  Loans with pre-loan tx:   {(pre_counts > 0).sum()}")
    print(f"  Min: {pre_counts.min()}, Median: {pre_counts.median()}, Mean: {pre_counts.mean():.1f}, Max: {pre_counts.max()}")

    # Check account opening date to loan date gap
    acc_loan_gap = (loan['loan_date'] - loan.merge(account, on='account_id')['account_date']).dt.days
    print(f"\nAccount age at loan disbursement (days):")
    print(f"  Min: {acc_loan_gap.min()} days, Median: {acc_loan_gap.median()} days, Mean: {acc_loan_gap.mean():.1f} days, Max: {acc_loan_gap.max()} days")

    # Leakage Reproduction:
    print("\n--- 7. Reproducing Documented Leakage Shortcuts ---")
    # Shortcut 1: Minimum balance over FULL lifetime of the account/loan
    # Calculate min balance per account across all transactions
    full_min_bal = trans.groupby('account_id')['balance'].min().reset_index()
    full_min_bal.columns = ['account_id', 'full_lifetime_min_balance']
    loan_with_minbal = loan.merge(full_min_bal, on='account_id', how='left')
    
    print("Summary of full lifetime min balance by status:")
    print(loan_with_minbal.groupby('status')['full_lifetime_min_balance'].describe())
    
    # Check simple threshold rule: full_lifetime_min_balance < 0 vs >= 0
    # For completed loans (A vs B):
    comp_leak = loan_with_minbal[loan_with_minbal['status'].isin(['A', 'B'])]
    pred_comp_bad = comp_leak['full_lifetime_min_balance'] < 0
    actual_comp_bad = comp_leak['status'] == 'B'
    comp_acc = (pred_comp_bad == actual_comp_bad).mean() * 100
    print(f"\nShortcut 1 on Completed Loans (A vs B): min_bal < 0 => Bad:")
    print(f"  Accuracy: {comp_acc:.2f}% ({((pred_comp_bad == actual_comp_bad).sum())} / {len(comp_leak)})")
    
    # For All Loans (A+C vs B+D):
    pred_all_bad = loan_with_minbal['full_lifetime_min_balance'] < 0
    actual_all_bad = loan_with_minbal['status'].isin(['B', 'D'])
    all_acc = (pred_all_bad == actual_all_bad).mean() * 100
    print(f"Shortcut 1 on All Loans (A+C vs B+D): min_bal < 0 => Bad:")
    print(f"  Accuracy: {all_acc:.2f}% ({((pred_all_bad == actual_all_bad).sum())} / {len(loan_with_minbal)})")
    
    # Let's check cross-tab of min_bal < 0 by status
    print("\nCrosstab of (full_lifetime_min_balance < 0) vs status:")
    print(pd.crosstab(loan_with_minbal['status'], loan_with_minbal['full_lifetime_min_balance'] < 0))

    # Shortcut 2: Duration shortcut or duration-based features reported by previous reviewer (~99.4%)
    # Let's inspect what duration vs loan date vs end of dataset reveals
    # The dataset ends in 1998-12-31.
    # What is loan_date + duration?
    # In loan.asc, duration is in months: 12, 24, 36, 48, 60.
    loan['end_date_approx'] = loan['loan_date'] + loan['duration'].apply(lambda m: pd.DateOffset(months=m))
    print("\nEnd date of loans (loan_date + duration months):")
    print(loan.groupby('status')['end_date_approx'].agg(['min', 'max']))
    dataset_end = trans['trans_date'].max()
    print(f"Dataset end date in trans: {dataset_end.date()}")
    
    # Are all A and B loans ended before dataset_end?
    print(f"Completed loans (A, B) end date <= dataset_end:")
    print((loan[loan['status'].isin(['A', 'B'])]['end_date_approx'] <= dataset_end).value_counts())
    print(f"Running loans (C, D) end date > dataset_end:")
    print((loan[loan['status'].isin(['C', 'D'])]['end_date_approx'] > dataset_end).value_counts())
    
    # Let's investigate the 99.4% duration shortcut reported by the reviewer.
    # Could it be predicting completed vs running, or predicting default?
    # Let's test combinations of duration, loan_date, etc.
    

if __name__ == '__main__':
    run_audit()
