"""
Clean Deterministic Data Ingestion & Schema Validation
Preserves raw files untouched and validates relational schemas and foreign keys.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

DEFAULT_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'berka-dataset')


def parse_yymmdd(val) -> pd.Timestamp:
    """Parse YYMMDD integer/string into pd.Timestamp with 1900-offset."""
    if pd.isna(val):
        return pd.NaT
    s = str(int(val)).zfill(6)
    yy = int(s[0:2])
    mm = int(s[2:4])
    dd = int(s[4:6])
    return pd.Timestamp(year=1900 + yy, month=mm, day=dd)


def parse_birth_number(val):
    """
    Parse Czech birth number:
    YYMMDD for men, YY(MM+50)DD for women.
    Returns: (birth_date: pd.Timestamp, gender: str)
    """
    s = str(int(val)).zfill(6)
    yy = int(s[0:2])
    mm_raw = int(s[2:4])
    dd = int(s[4:6])
    if mm_raw > 50:
        gender = 'F'
        mm = mm_raw - 50
    else:
        gender = 'M'
        mm = mm_raw
    bdate = pd.Timestamp(year=1900 + yy, month=mm, day=dd)
    return bdate, gender


class BerkaDataLoader:
    def __init__(self, raw_dir: str = DEFAULT_RAW_DIR):
        self.raw_dir = raw_dir
        self._validate_raw_files_exist()

    def _validate_raw_files_exist(self):
        expected_files = [
            'loan.asc', 'account.asc', 'disp.asc', 'client.asc',
            'card.asc', 'district.asc', 'order.asc', 'trans.asc'
        ]
        for f in expected_files:
            p = os.path.join(self.raw_dir, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Raw data file not found: {p}")

    def load_loans(self) -> pd.DataFrame:
        p = os.path.join(self.raw_dir, 'loan.asc')
        df = pd.read_csv(p, sep=';')
        assert len(df) == 682, f"Expected 682 loans, got {len(df)}"
        assert df['loan_id'].is_unique, "loan_id must be unique primary key"
        assert df['account_id'].is_unique, "Each loan must map to a unique account_id"
        df['loan_date'] = df['date'].apply(parse_yymmdd)
        df['duration_months'] = df['duration'].astype(int)
        df['amount'] = df['amount'].astype(float)
        df['payments'] = df['payments'].astype(float)
        df['status'] = df['status'].str.strip()
        return df

    def load_accounts(self) -> pd.DataFrame:
        p = os.path.join(self.raw_dir, 'account.asc')
        df = pd.read_csv(p, sep=';')
        assert len(df) == 4500, f"Expected 4500 accounts, got {len(df)}"
        assert df['account_id'].is_unique, "account_id must be unique primary key"
        df['account_date'] = df['date'].apply(parse_yymmdd)
        df['frequency'] = df['frequency'].str.strip()
        return df

    def load_dispositions(self) -> pd.DataFrame:
        p = os.path.join(self.raw_dir, 'disp.asc')
        df = pd.read_csv(p, sep=';')
        assert len(df) == 5369, f"Expected 5369 dispositions, got {len(df)}"
        assert df['disp_id'].is_unique, "disp_id must be unique primary key"
        df['type'] = df['type'].str.strip()
        return df

    def load_clients(self) -> pd.DataFrame:
        p = os.path.join(self.raw_dir, 'client.asc')
        df = pd.read_csv(p, sep=';')
        assert len(df) == 5369, f"Expected 5369 clients, got {len(df)}"
        assert df['client_id'].is_unique, "client_id must be unique primary key"
        parsed = df['birth_number'].apply(parse_birth_number)
        df['birth_date'] = [p[0] for p in parsed]
        df['gender'] = [p[1] for p in parsed]
        return df

    def load_districts(self) -> pd.DataFrame:
        p = os.path.join(self.raw_dir, 'district.asc')
        df = pd.read_csv(p, sep=';')
        assert len(df) == 77, f"Expected 77 districts, got {len(df)}"
        df = df.rename(columns={
            'A1': 'district_id', 'A2': 'district_name', 'A3': 'region',
            'A4': 'inhabitants', 'A5': 'mun_lt_499', 'A6': 'mun_500_1999',
            'A7': 'mun_2000_9999', 'A8': 'mun_gt_10000', 'A9': 'cities',
            'A10': 'urban_ratio', 'A11': 'avg_salary',
            'A12': 'unemploy_95', 'A13': 'unemploy_96',
            'A14': 'entrepreneurs_per_1000',
            'A15': 'crimes_95', 'A16': 'crimes_96'
        })
        # Clean '?' in unemploy_95 (A12) and crimes_95 (A15)
        for col in ['unemploy_95', 'unemploy_96', 'crimes_95', 'crimes_96']:
            df[col] = pd.to_numeric(df[col].replace('?', np.nan), errors='coerce')
        # Impute district 69 (Jesenik) missing 1995 values using 1996 values or region medians
        df['unemploy_95'] = df['unemploy_95'].fillna(df['unemploy_96'])
        df['crimes_95'] = df['crimes_95'].fillna(df['crimes_96'])
        return df

    def load_transactions(self) -> pd.DataFrame:
        p = os.path.join(self.raw_dir, 'trans.asc')
        df = pd.read_csv(p, sep=';', low_memory=False)
        assert len(df) == 1056320, f"Expected 1056320 transactions, got {len(df)}"
        assert df['trans_id'].is_unique, "trans_id must be unique primary key"
        df['trans_date'] = df['date'].apply(parse_yymmdd)
        df['amount'] = df['amount'].astype(float)
        df['balance'] = df['balance'].astype(float)
        df['type'] = df['type'].str.strip()
        df['operation'] = df['operation'].astype(str).str.strip().replace({'nan': np.nan, '': np.nan})
        df['k_symbol'] = df['k_symbol'].astype(str).str.strip().replace({'nan': np.nan, '': np.nan})
        return df

    def build_master_loan_entities(self) -> pd.DataFrame:
        """
        Build master entity table for all 682 loans, joining account,
        owner disposition, client, and district data.
        Guarantees 1 row per loan with 0 duplication.
        """
        loans = self.load_loans()
        accounts = self.load_accounts()
        disps = self.load_dispositions()
        clients = self.load_clients()
        districts = self.load_districts()

        # Filter dispositions to OWNER (only owner can take a loan)
        owner_disps = disps[disps['type'] == 'OWNER']
        assert len(owner_disps) == 4500, "Expected 4500 owner dispositions"

        # Merge loan with account
        m = loans.merge(accounts, on='account_id', how='inner', suffixes=('', '_acc'))
        assert len(m) == 682, "Merge loan-account must retain exactly 682 rows"

        # Merge with owner disposition
        m = m.merge(owner_disps[['disp_id', 'client_id', 'account_id']], on='account_id', how='inner')
        assert len(m) == 682, "Merge with owner disposition must retain exactly 682 rows"

        # Merge with client
        m = m.merge(clients, on='client_id', how='inner', suffixes=('', '_client'))
        assert len(m) == 682, "Merge with client must retain exactly 682 rows"

        # Merge with district (account branch location)
        m = m.merge(districts, on='district_id', how='inner')
        assert len(m) == 682, "Merge with district must retain exactly 682 rows"

        # Add pre-computed relationship age at loan disbursement
        m['account_age_days_at_loan'] = (m['loan_date'] - m['account_date']).dt.days
        m['client_age_years_at_loan'] = (m['loan_date'] - m['birth_date']).dt.days / 365.25

        return m


if __name__ == '__main__':
    loader = BerkaDataLoader()
    master = loader.build_master_loan_entities()
    print(f"Master loan entities loaded successfully: {master.shape}")
    print(master[['loan_id', 'account_id', 'client_id', 'loan_date', 'amount', 'duration_months', 'status']].head())
