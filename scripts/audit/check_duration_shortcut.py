import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'berka-dataset')

def parse_date(val):
    s = str(val).strip()
    return pd.Timestamp(year=1900 + int(s[0:2]), month=int(s[2:4]), day=int(s[4:6]))

loan = pd.read_csv(os.path.join(DATA_DIR, 'loan.asc'), sep=';')
loan['loan_date'] = loan['date'].apply(parse_date)

# Test approx duration in days (e.g. duration * 30 days)
loan['end_approx_30'] = loan['loan_date'] + pd.to_timedelta(loan['duration'] * 30, unit='D')
cutoff = pd.Timestamp('1998-12-31')
pred_completed_approx = loan['end_approx_30'] <= cutoff
actual_completed = loan['status'].isin(['A', 'B'])

acc_approx = (pred_completed_approx == actual_completed).mean() * 100
print(f"Accuracy of (loan_date + duration*30 <= 1998-12-31) predicting Completed (A,B) vs Running (C,D): {acc_approx:.2f}%")
print(f"Mismatches: {(pred_completed_approx != actual_completed).sum()} out of {len(loan)}")

# What about predicting default using duration?
for d in sorted(loan['duration'].unique()):
    sub = loan[loan['duration'] == d]
    bad = sub['status'].isin(['B', 'D']).sum()
    print(f"Duration {d} months: Total={len(sub)}, Bad={bad} ({bad/len(sub)*100:.2f}%)")

# What about amount, payments, ratio of amount / payments?
loan['implied_duration'] = loan['amount'] / loan['payments']
print("Implied duration (amount / payments) summary:")
print(loan[['duration', 'implied_duration']].head(10))
print("Correlation between duration and implied_duration:", loan['duration'].corr(loan['implied_duration']))
