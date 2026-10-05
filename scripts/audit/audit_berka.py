import os
import sys
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'berka-dataset')

def inspect_raw_files():
    print("=== 1. RAW FILES AND SCHEMA INSPECTION ===")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.asc')]
    for fname in sorted(files):
        fpath = os.path.join(DATA_DIR, fname)
        size = os.path.getsize(fpath)
        with open(fpath, 'r', encoding='latin1') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            # Count lines
            f.seek(0)
            line_count = sum(1 for _ in f) - 1 # header excluded
        print(f"File: {fname:<15} Size: {size:>10} bytes | Rows (excl header): {line_count:>8} | Header: {first_line[:70]}")

if __name__ == '__main__':
    inspect_raw_files()
