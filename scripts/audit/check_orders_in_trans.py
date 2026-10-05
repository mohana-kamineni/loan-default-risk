import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'berka-dataset')

def parse_date(val):
    s = str(val).strip()
    return pd.Timestamp(year=1900 + int(s[0:2]), month=int(s[2:4]), day=int(s[4:6]))

loan = pd.read_csv(os.path.join(DATA_DIR, 'loan.asc'), sep=';')
order = pd.read_csv(os.path.join(DATA_DIR, 'order.asc'), sep=';')
trans = pd.read_csv(os.path.join(DATA_DIR, 'trans.asc'), sep=';', low_memory=False)

loan['loan_date'] = loan['date'].apply(parse_date)
trans['trans_date'] = trans['date'].apply(parse_date)

# In trans.asc, what operations correspond to permanent orders?
# The PKDD'99 documentation says: "PREVOD NA UCET remittance to another bank"
# Let's inspect trans where operation == 'PREVOD NA UCET' in early windows
early_trans = loan[['loan_id', 'account_id', 'loan_date']].merge(trans, on='account_id')
early_trans['days'] = (early_trans['trans_date'] - early_trans['loan_date']).dt.days

for w in [30, 60, 90]:
    w_trans = early_trans[(early_trans['days'] >= 0) & (early_trans['days'] <= w)]
    remit = w_trans[w_trans['operation'] == 'PREVOD NA UCET']
    print(f"Window [0, {w}] days: total remittance debits (PREVOD NA UCET): {len(remit)}")
    print(f"  Remittance k_symbols:\n{remit['k_symbol'].value_counts(dropna=False)}")
