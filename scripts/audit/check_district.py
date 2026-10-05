import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'berka-dataset')
district = pd.read_csv(os.path.join(DATA_DIR, 'district.asc'), sep=';')
account = pd.read_csv(os.path.join(DATA_DIR, 'account.asc'), sep=';')
loan = pd.read_csv(os.path.join(DATA_DIR, 'loan.asc'), sep=';')

print("District shape:", district.shape)
print("District head:\n", district.head(2))

# District mapping with loan accounts
loan_acc = loan.merge(account, on='account_id')
print("Unique districts among loan accounts:", loan_acc['district_id'].nunique())
print("Default rate by district distribution:")
loan_acc['is_bad'] = loan_acc['status'].isin(['B', 'D'])
dist_stats = loan_acc.groupby('district_id').agg(
    n_loans=('loan_id', 'count'),
    n_bad=('is_bad', 'sum')
)
dist_stats['bad_rate'] = dist_stats['n_bad'] / dist_stats['n_loans']
print(dist_stats.describe())
print("Prague district (district_id == 1):")
print(dist_stats.loc[1] if 1 in dist_stats.index else "Not found")
