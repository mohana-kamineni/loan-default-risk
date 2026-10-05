import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'berka-dataset')

def parse_date(val):
    s = str(val).strip()
    return pd.Timestamp(year=1900 + int(s[0:2]), month=int(s[2:4]), day=int(s[4:6]))

loan = pd.read_csv(os.path.join(DATA_DIR, 'loan.asc'), sep=';')
trans = pd.read_csv(os.path.join(DATA_DIR, 'trans.asc'), sep=';', low_memory=False)
loan['loan_date'] = loan['date'].apply(parse_date)
trans['trans_date'] = trans['date'].apply(parse_date)

# 1. Inspect status A loan with SANKC. UROK
merged = loan.merge(trans, on='account_id')
merged['days'] = (merged['trans_date'] - merged['loan_date']).dt.days
a_sankc = merged[(merged['status'] == 'A') & (merged['k_symbol'] == 'SANKC. UROK')]
print("Status A loans with SANKC. UROK:")
print(a_sankc[['loan_id', 'account_id', 'status', 'trans_date', 'loan_date', 'days', 'amount_y', 'balance', 'k_symbol']])

# Inspect all transactions of this loan's account around that time
if len(a_sankc) > 0:
    acc_id = a_sankc['account_id'].iloc[0]
    sub_trans = trans[trans['account_id'] == acc_id].sort_values('trans_date')
    print(f"\nAll transactions for account {acc_id} around that date:")
    print(sub_trans[(sub_trans['trans_date'] >= '1995-01-01') & (sub_trans['trans_date'] <= '1995-06-30')][['trans_date', 'type', 'operation', 'amount', 'balance', 'k_symbol']])

# 2. Inspect negative balances before loan date
pre_neg = merged[(merged['days'] < 0) & (merged['balance'] < 0)]
print(f"\nTotal pre-loan transactions with negative balance: {len(pre_neg)}")
print(f"Unique loans with pre-loan negative balance: {pre_neg['loan_id'].nunique()}")
print("Status breakdown of loans with pre-loan negative balance:")
print(loan[loan['loan_id'].isin(pre_neg['loan_id'])]['status'].value_counts())
