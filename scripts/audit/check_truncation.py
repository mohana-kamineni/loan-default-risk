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

dataset_end = trans['trans_date'].max()
print(f"Dataset end date: {dataset_end.date()}")

# For each loan, what is the maximum available observation time before dataset end?
loan['days_until_dataset_end'] = (dataset_end - loan['loan_date']).dt.days

print("\nLoans with fewer days until dataset end than the observation window:")
for w in [30, 60, 90]:
    truncated = loan[loan['days_until_dataset_end'] < w]
    print(f"Window {w:2d} days: {len(truncated):2d} loans have fewer than {w} days of potential observation before dataset cutoff!")
    if len(truncated) > 0:
        print(f"  Truncated loan statuses: {truncated['status'].value_counts().to_dict()}")
        print(f"  Loan dates of truncated: {truncated['loan_date'].min().date()} to {truncated['loan_date'].max().date()}")

# Let's inspect the claim by previous reviewer:
# "All 682 loans have at least 3 transactions within 90 days of loan disbursement."
# Let's check how many transactions each loan actually has within [loan_date, loan_date + 90 days]
# AND how many days between loan_date and the LAST transaction of that account!
acc_last_trans = trans.groupby('account_id')['trans_date'].max().reset_index()
acc_last_trans.columns = ['account_id', 'last_trans_date']
loan = loan.merge(acc_last_trans, on='account_id', how='left')
loan['trans_history_days'] = (loan['last_trans_date'] - loan['loan_date']).dt.days

print("\nTransaction history days after loan date:")
for w in [30, 60, 90]:
    short_hist = loan[loan['trans_history_days'] < w]
    print(f"Loans where last transaction is < {w} days after loan: {len(short_hist)}")
    if len(short_hist) > 0:
        print(short_hist[['loan_id', 'status', 'loan_date', 'last_trans_date', 'trans_history_days']])
