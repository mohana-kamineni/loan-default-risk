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

# Check transactions around loan date for several accounts
merged = loan.merge(trans, on='account_id')
merged['days'] = (merged['trans_date'] - merged['loan_date']).dt.days

# Check whether transaction amount matches loan amount
loan_disburse_match = merged[(merged['days'].isin([0, 1, -1])) & (merged['amount_y'] == merged['amount_x'])]
print(f"Transactions on day -1, 0, or +1 with amount matching loan amount: {len(loan_disburse_match)}")

# Check transactions on day 0 specifically
day0 = merged[merged['days'] == 0]
print(f"Total transactions on day 0: {len(day0)}")
print("Summary of day 0 transactions by type and operation:")
print(day0.groupby(['type', 'operation'])['amount_y'].describe())

# Check first transaction AFTER loan date (days > 0)
post_loan = merged[merged['days'] > 0]
first_post = post_loan.groupby('loan_id').agg(
    first_days=('days', 'min'),
    first_type=('type', 'first'),
    first_op=('operation', 'first'),
    first_k=('k_symbol', 'first'),
    first_amt=('amount_y', 'first')
).reset_index()

print("\nDays to FIRST post-loan transaction (days > 0):")
print(first_post['first_days'].describe())

# How many days between loan date and first post-loan transaction?
print("Distribution of days to first post-loan transaction:")
print(pd.cut(first_post['first_days'], bins=[0, 1, 5, 10, 15, 30, 60, 100]).value_counts().sort_index())
