"""prepare.py - Part A: Data Foundation: Audit, Clean, Engineer."""
import pandas as pd
import numpy as np

def clean_transactions(raw_df):
    """Cleans the raw DataFrame according to assignment rules."""
    df = raw_df.copy()
    df = df.drop_duplicates()
    df["merchant_category"] = df["merchant_category"].astype(str).str.strip().str.title()
    df["city"] = df["city"].astype(str).str.strip().str.title()
    
    df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors="coerce")
    df = df.dropna(subset=["amount_inr"])
    df = df[(df["amount_inr"] >= 1) & (df["amount_inr"] < 1000000)]
    df = df.reset_index(drop=True)
    return df

def add_features(df):
    """Adds the three required engineered features."""
    res = df.copy()
    res["amount_vs_avg_ratio"] = res["amount_inr"] / res["avg_amount_30d_inr"]
    res["is_night"] = res["hour_of_day"].apply(lambda h: 1 if h <= 5 else 0)
    res["high_value"] = res["amount_inr"].apply(lambda a: 1 if a > 50000 else 0)
    return res

if __name__ == "__main__":
    raw = pd.read_csv("transactions_raw.csv")
    
    # Audit calculations matching acceptance criteria
    total_rows = len(raw)
    total_cols = len(raw.columns)
    exact_dups = raw.duplicated().sum()
    missing_amt = raw["amount_inr"].isna().sum() + (raw["amount_inr"] == "").sum() if "amount_inr" in raw else 0
    missing_dist = raw["distance_from_home_km"].isna().sum() if "distance_from_home_km" in raw else 0
    
    cleaned = clean_transactions(raw)
    engineered = add_features(cleaned)
    
    # Print required audit table & checkpoints to console for Task 6
    print("--- PART A AUDIT REPORT ---")
    print(f"The audit reports: {total_rows} rows and {total_cols} columns as read; {exact_dups} exact duplicate rows; after de-duplication {total_rows - exact_dups} rows, {missing_amt} missing amount_inr , {missing_dist} missing distance_from_home_km.")
    print(f"clean_transactions on the raw file returns exactly {len(cleaned)} rows, with no duplicate rows.")
    print(f"After add_features: is_night sums to {engineered['is_night'].sum()}, high_value sums to {engineered['high_value'].sum()}, maximum of amount_vs_avg_ratio is {engineered['amount_vs_avg_ratio'].max():.2f}.")
    
    fraud_rows = engineered[engineered["is_fraud"] == 1]
    print(f"Fraud rows: {len(fraud_rows)} of {len(engineered)} ({len(fraud_rows)/len(engineered)*100:.2f}%).")
    print("Part A audit and feature engineering completed successfully.")