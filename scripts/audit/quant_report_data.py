import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'berka-dataset')

def parse_date(val):
    s = str(val).strip()
    return pd.Timestamp(year=1900 + int(s[0:2]), month=int(s[2:4]), day=int(s[4:6]))

loan = pd.read_csv(os.path.join(DATA_DIR, 'loan.asc'), sep=';')
trans = pd.read_csv(os.path.join(DATA_DIR, 'trans.asc'), sep=';', low_memory=False)

loan['loan_date'] = loan['date'].apply(parse_date)
trans['trans_date'] = trans['date'].apply(parse_date)

merged = loan.merge(trans, on='account_id')
merged['days'] = (merged['trans_date'] - merged['loan_date']).dt.days

print("=== QUANTITATIVE BREAKDOWN FOR DESIGN DECISION REPORT ===")

for w in [30, 60, 90]:
    print(f"\n--- WINDOW: {w} DAYS ---")
    w_trans = merged[(merged['days'] >= 0) & (merged['days'] <= w)]
    
    # Per loan stats
    tx_counts = w_trans.groupby('loan_id').size().reindex(loan['loan_id'], fill_value=0)
    min_bals = w_trans.groupby('loan_id')['balance'].min().reindex(loan['loan_id'], fill_value=np.nan)
    
    # Check default status
    temp = loan[['loan_id', 'status']].copy()
    temp['tx_count'] = tx_counts.values
    temp['min_bal'] = min_bals.values
    temp['went_neg'] = temp['min_bal'] < 0
    temp['is_bad'] = temp['status'].isin(['B', 'D'])
    
    print(f"Overall Transaction Counts in [0, {w}]:")
    print(f"  Mean tx: {tx_counts.mean():.2f} | Median: {tx_counts.median():.1f} | Min: {tx_counts.min()} | Max: {tx_counts.max()}")
    print(f"  Loans with < 3 tx: {(tx_counts < 3).sum()} | < 5 tx: {(tx_counts < 5).sum()}")
    
    print(f"Negative balance in window [0, {w}]:")
    print(pd.crosstab(temp['status'], temp['went_neg'], margins=True))
    
    # For bad loans: how many went negative within window vs will go negative later?
    bad_temp = temp[temp['is_bad']]
    neg_in_w = bad_temp['went_neg'].sum()
    future_neg = len(bad_temp) - neg_in_w
    print(f"  Among 76 bad loans: {neg_in_w} ({neg_in_w/76*100:.1f}%) already negative within {w}d; {future_neg} ({future_neg/76*100:.1f}%) still solvent at day {w}")
    
    # For completed loans (A vs B):
    comp_temp = temp[temp['status'].isin(['A', 'B'])]
    b_temp = comp_temp[comp_temp['status'] == 'B']
    b_neg_in_w = b_temp['went_neg'].sum()
    b_future_neg = len(b_temp) - b_neg_in_w
    print(f"  Among 31 B loans: {b_neg_in_w} ({b_neg_in_w/31*100:.1f}%) already negative within {w}d; {b_future_neg} ({b_future_neg/31*100:.1f}%) still solvent at day {w}")

