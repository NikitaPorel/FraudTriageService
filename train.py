"""train.py - Part B: Fraud Model, Imbalance and Alert Threshold."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import prepare

# Load and prepare cleaned data with features
raw_df = pd.read_csv("transactions_raw.csv")
clean_df = prepare.clean_transactions(raw_df)
df = prepare.add_features(clean_df)

# Safety check: ensure engineered features exist even if prepare.py missed them
if 'high_value' not in df.columns:
    df['high_value'] = (df['amount_inr'] > 50000).astype(int)
if 'amount_vs_avg_ratio' not in df.columns:
    df['amount_vs_avg_ratio'] = df['amount_inr'] / df['avg_amount_30d_inr']
if 'is_night' not in df.columns:
    df['is_night'] = (df['hour_of_day'] <= 5).astype(int)

# Exclude identifier/target columns and handle categorical encoding
X_raw = df.drop(columns=["txn_id", "customer_id", "chargeback_filed", "is_fraud"])
y = df["is_fraud"]

# Categorical columns to one-hot encode (categorical levels are fixed and known)
X = pd.get_dummies(X_raw, dtype=int)

# Pin the input column order exactly as required
pinned_numeric_binary = [
    "amount_inr", "hour_of_day", "txn_count_last_24h", "avg_amount_30d_inr",
    "distance_from_home_km", "is_new_device", "account_age_days",
    "amount_vs_avg_ratio", "is_night", "high_value"
]

# Extract category dummies in alphabetical order per category
merchant_cols = sorted([c for c in X.columns if c.startswith("merchant_category_")])
city_cols = sorted([c for c in X.columns if c.startswith("city_")])
payment_cols = sorted([c for c in X.columns if c.startswith("payment_mode_")])

pinned_columns = pinned_numeric_binary + merchant_cols + city_cols + payment_cols
X = X[pinned_columns]

print("Pinned Columns List (26 input columns):")
print(list(X.columns))

# 1. Stratified 60/20/20 Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

train_median_dist = X_train["distance_from_home_km"].median()

X_train["distance_from_home_km"] = X_train["distance_from_home_km"].fillna(train_median_dist)
X_val["distance_from_home_km"] = X_val["distance_from_home_km"].fillna(train_median_dist)
X_test["distance_from_home_km"] = X_test["distance_from_home_km"].fillna(train_median_dist)

# 2. Three Ways to Treat Imbalance (Logistic Regression)
def eval_lr_model(model, X_train_data, y_train_data, X_val_data, y_val_data):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    pipe.fit(X_train_data, y_train_data)
    y_pred_prob = pipe.predict_proba(X_val_data)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    prec = precision_score(y_val_data, y_pred, zero_division=0)
    rec = recall_score(y_val_data, y_pred, zero_division=0)
    f1 = f1_score(y_val_data, y_pred, zero_division=0)
    cm = confusion_matrix(y_val_data, y_pred)
    return prec, rec, f1, cm

lr_plain = LogisticRegression(max_iter=1000, random_state=42)
p1, r1, f1_1, cm1 = eval_lr_model(lr_plain, X_train, y_train, X_val, y_val)

lr_balanced = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
p2, r2, f1_2, cm2 = eval_lr_model(lr_balanced, X_train, y_train, X_val, y_val)

non_fraud_train = X_train[y_train == 0]
fraud_train = X_train[y_train == 1]
fraud_oversampled = fraud_train.sample(n=len(non_fraud_train), replace=True, random_state=42)

X_train_os = pd.concat([non_fraud_train, fraud_oversampled], axis=0)
y_train_os = np.array([0]*len(non_fraud_train) + [1]*len(fraud_oversampled))

p3, r3, f1_3, cm3 = eval_lr_model(lr_plain, X_train_os, y_train_os, X_val, y_val)

print("\n--- Imbalance Treatment Evaluation (Logistic Regression @ prob >= 0.5) ---")
print(f"Plain         -> Precision: {p1:.3f}, Recall: {r1:.3f}, F1: {f1_1:.3f}, CM: {cm1.ravel()}")
print(f"Balanced      -> Precision: {p2:.3f}, Recall: {r2:.3f}, F1: {f1_2:.3f}, CM: {cm2.ravel()}")
print(f"Over-sampled  -> Precision: {p3:.3f}, Recall: {r3:.3f}, F1: {f1_3:.3f}, CM: {cm3.ravel()}")

# 3. Cross-Validation and Tuning (Random Forest)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'max_depth': [4, 6, 8, None],
    'min_samples_leaf': [5, 10, 20]
}

rf_base = RandomForestClassifier(
    n_estimators=200, 
    class_weight='balanced', 
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    scoring='f1',
    cv=skf
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print("\n--- Random Forest Tuning Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Mean CV F1 Score: {grid_search.best_score_:.4f}")

# 4. Threshold Optimization on Validation Split
val_probs = best_rf.predict_proba(X_val)[:, 1]
thresholds = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]
t_star = 0.1
max_t_valid = 0.1

print("\n--- Threshold Analysis on Validation Set ---")
for t in thresholds:
    preds = (val_probs >= t).astype(int)
    p = precision_score(y_val, preds, zero_division=0)
    r = recall_score(y_val, preds, zero_division=0)
    f = f1_score(y_val, preds, zero_division=0)
    flagged = np.sum(preds)
    print(f"Threshold: {t:.1f} | Precision: {p:.3f} | Recall: {r:.3f} | F1: {f:.3f} | Flagged: {flagged}")
    
    if r >= 0.65 and t > max_t_valid:
        max_t_valid = t
        t_star = t

print(f"\nSelected t_star: {t_star}")

# 5. One Final Test Evaluation
test_probs = best_rf.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= t_star).astype(int)

test_prec = precision_score(y_test, test_preds, zero_division=0)
test_rec = recall_score(y_test, test_preds, zero_division=0)
test_f1 = f1_score(y_test, test_preds, zero_division=0)
test_cm = confusion_matrix(y_test, test_preds)

print("\n--- Final Test Set Results ---")
print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1: {test_f1:.3f}")
print(f"Confusion Matrix (TN, FP, FN, TP): {test_cm.ravel()}")
print(f"Flagged Payments: {np.sum(test_preds)}")

# 6. Leakage Experiment
X_leaky = pd.concat([X, df[["chargeback_filed"]]], axis=1)

X_tr_leak, X_temp_leak, y_tr_leak, y_temp_leak = train_test_split(
    X_leaky, y, test_size=0.4, stratify=y, random_state=42
)
X_val_leak, _, y_val_leak, _ = train_test_split(
    X_temp_leak, y_temp_leak, test_size=0.5, stratify=y_temp_leak, random_state=42
)

X_tr_leak["distance_from_home_km"] = X_tr_leak["distance_from_home_km"].fillna(train_median_dist)
X_val_leak["distance_from_home_km"] = X_val_leak["distance_from_home_km"].fillna(train_median_dist)

leaky_rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    **grid_search.best_params_
)
leaky_rf.fit(X_tr_leak, y_tr_leak)

leaky_val_probs = leaky_rf.predict_proba(X_val_leak)[:, 1]
leaky_val_preds = (leaky_val_probs >= 0.5).astype(int)
leaky_f1 = f1_score(y_val_leak, leaky_val_preds, zero_division=0)

clean_val_preds = (val_probs >= 0.5).astype(int)
clean_f1 = f1_score(y_val, clean_val_preds, zero_division=0)

print("\n--- Leakage Experiment Results ---")
print(f"Clean Forest Validation F1 (@0.5): {clean_f1:.4f}")
print(f"Leaky Forest Validation F1 (@0.5): {leaky_f1:.4f}")

# 7. Feature Importance Analysis
importances = best_rf.feature_importances_
feature_names = list(X.columns)
top5_indices = np.argsort(importances)[::-1][:5]

print("\n--- Top 5 Important Features ---")
for idx in top5_indices:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")

# 8. Reusable Builder Function
def build_final_model():
    raw_data = pd.read_csv("transactions_raw.csv")
    c_df = prepare.clean_transactions(raw_data)
    f_df = prepare.add_features(c_df)
    
    # Safety check inside builder function
    if 'high_value' not in f_df.columns:
        f_df['high_value'] = (f_df['amount_inr'] > 50000).astype(int)
    if 'amount_vs_avg_ratio' not in f_df.columns:
        f_df['amount_vs_avg_ratio'] = f_df['amount_inr'] / f_df['avg_amount_30d_inr']
    if 'is_night' not in f_df.columns:
        f_df['is_night'] = (f_df['hour_of_day'] <= 5).astype(int)
        
    X_all = f_df.drop(columns=["txn_id", "customer_id", "chargeback_filed", "is_fraud"])
    y_all = f_df["is_fraud"]
    
    X_enc = pd.get_dummies(X_all, dtype=int)
    X_enc = X_enc[pinned_columns]
    
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_enc, y_all, test_size=0.4, stratify=y_all, random_state=42
    )
    X_v, X_te, y_v, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )
    
    med = X_tr["distance_from_home_km"].median()
    X_tr["distance_from_home_km"] = X_tr["distance_from_home_km"].fillna(med)
    X_v["distance_from_home_km"] = X_v["distance_from_home_km"].fillna(med)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        **grid_search.best_params_
    )
    rf.fit(X_tr, y_tr)
    
    v_probs = rf.predict_proba(X_v)[:, 1]
    t_opt = 0.1
    max_t = 0.1
    for t in [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]:
        r = recall_score(y_v, (v_probs >= t).astype(int), zero_division=0)
        if r >= 0.65 and t > max_t:
            max_t = t
            t_opt = t
            
    return {
        "model": rf,
        "columns": pinned_columns,
        "median": med,
        "threshold": t_opt
    }

if __name__ == "__main__":
    pass