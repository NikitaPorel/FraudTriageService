--- PART A AUDIT REPORT ---
The audit reports: 6090 rows and 14 columns as read; 90 exact duplicate rows; after de-duplication 6000 rows, 79 missing amount_inr , 196missing distance_from_home_km.
clean_transactions on the raw file returns exactly 5919 rows, with no duplicate rows.
After add_features: is_night sums to 365, high_value sums to 17, maximum of amount_vs_avg_ratio is 13.11.
Fraud rows: 366 of 5919 (6.18%).
Part A audit and feature engineering completed successfully.




--- PART B Result ---
Pinned Columns List (26 input columns):
['amount_inr', 'hour_of_day', 'txn_count_last_24h', 'avg_amount_30d_inr', 'distance_from_home_km', 'is_new_device', 'account_age_days', 'amount_vs_avg_ratio', 'is_night', 'high_value', 'merchant_category_Electronics', 'merchant_category_Fashion', 'merchant_category_Food Delivery', 'merchant_category_Grocery', 'merchant_category_Travel', 'merchant_category_Utilities', 'city_Bengaluru', 'city_Chennai', 'city_Delhi', 'city_Kolkata', 'city_Mumbai', 'city_Pune', 'payment_mode_Card','payment_mode_NetBanking', 'payment_mode_UPI', 'payment_mode_Wallet']

--- Imbalance Treatment Evaluation (Logistic Regression @ prob >= 0.5) ---
Plain         -> Precision: 0.625, Recall: 0.342, F1: 0.442, CM: [1096   15   48   25]
Balanced      -> Precision: 0.252, Recall: 0.781, F1: 0.381, CM: [942 169  16  57]
Over-sampled  -> Precision: 0.257, Recall: 0.781, F1: 0.386, CM: [946 165  16  57]

--- Random Forest Tuning Results ---
Best Parameters: {'max_depth': None, 'min_samples_leaf': 5}
Best Mean CV F1 Score: 0.4646

--- Threshold Analysis on Validation Set ---
Threshold: 0.1 | Precision: 0.111 | Recall: 0.973 | F1: 0.200 | Flagged: 638
Threshold: 0.2 | Precision: 0.158 | Recall: 0.945 | F1: 0.271 | Flagged: 437
Threshold: 0.3 | Precision: 0.209 | Recall: 0.863 | F1: 0.336 | Flagged: 302
Threshold: 0.4 | Precision: 0.272 | Recall: 0.712 | F1: 0.394 | Flagged: 191
Threshold: 0.5 | Precision: 0.352 | Recall: 0.589 | F1: 0.441 | Flagged: 122
Threshold: 0.6 | Precision: 0.507 | Recall: 0.507 | F1: 0.507 | Flagged: 73
Threshold: 0.7 | Precision: 0.528 | Recall: 0.260 | F1: 0.349 | Flagged: 36
Threshold: 0.8 | Precision: 0.778 | Recall: 0.192 | F1: 0.308 | Flagged: 18
Threshold: 0.9 | Precision: 0.800 | Recall: 0.055 | F1: 0.103 | Flagged: 5

Selected t_star: 0.4

--- Final Test Set Results ---
Precision: 0.281, Recall: 0.781, F1: 0.413
Confusion Matrix (TN, FP, FN, TP): [965 146  16  57]
Flagged Payments: 203

--- Leakage Experiment Results ---
Clean Forest Validation F1 (@0.5): 0.4410
Leaky Forest Validation F1 (@0.5): 0.7975

--- Top 5 Important Features ---
amount_vs_avg_ratio: 0.2700
txn_count_last_24h: 0.1779
amount_inr: 0.1339
account_age_days: 0.0835
distance_from_home_km: 0.0709




--- PART C Result 1 ---
Query                          | Expected   | Retrieved  | Score   
--------------------------------------------------------------------
card payment above INR 50000   | P1         | P1         | 0.5501
how many UPI payments in 24    | P2         | P2         | 0.5622
payment at 3 AM far away fro   | P3         | P3         | 0.4948
customer reports unauthorise   | P4         | P4         | 0.5689
electronics merchant payment   | P5         | P5         | 0.5641
which payments are approved    | P6         | P6         | 0.6250
temporary hold on repeated U   | P2         | P2         | 0.4862
what happens after a custome   | P4         | P4         | 0.4867
what is the cafeteria menu t   | None       | P3         | 0.4251
how do I reset my email pass   | None       | P1         | 0.0000

In-scope top-1 hit count: 8

--- PART C Result 2 ---
Scenario        | Retrieved  | Attempts | Failures   | Status | SafeDef | Final Action     | Override
-----------------------------------------------------------------------------------------------
S1              | P1         | 1        | []         | ok     | False    | hold_for_review  | False   
S2              | P2         | 2        | [1]        | ok     | False    | hold_for_review  | False   
S3              | P3         | 2        | [2]        | ok     | False    | hold_for_review  | False   
S4              | P4         | 2        | [3]        | ok     | False    | block_and_notify | False   
S5              | P5         | 3        | [1, 1, 1]  | ok     | True    | hold_for_review  | False   
S6              | P2         | 1        | []         | ok     | False    | hold_for_review  | True    
S7              | P1         | 2        | [3]        | ok     | False    | hold_for_review  | False   
S8              | P4         | 1        | []         | ok     | False    | block_and_notify | False   
S9              | P3         | 3        | [1, 2]     | ok     | False    | hold_for_review  | False   
S10             | P2         | 2        | [3]        | ok     | False    | hold_for_review  | False   





--- PART D Result ---
Waiting for service to start...
Service is up!
[Q1] HAPPY - PASS (Status: 200)
[Q2] HAPPY - PASS (Status: 200)
[Q3] ESCALATION - PASS (Status: 200)
[Q4] ESCALATION - PASS (Status: 200)
[Q5] EDGE - PASS (Status: 400)
[Q6] EDGE - PASS (Status: 422)
[Q7] ESCALATION - PASS (Status: 200)
[Q8] ESCALATION - PASS (Status: 200)
[Q9] EDGE - FAIL (Expected 503, got 200)
[Q10] HAPPY - PASS (Status: 200)
[Q11] EDGE - PASS (Status: 200)

--- Per-Category Pass Counts ---
happy: 3 passes
escalation: 4 passes
edge: 3 passes

Total Passes: 10 / 11

--- Metrics JSON Response ---
{
  "requests_total": 10,
  "success_total": 8,
  "error_total": 2,
  "success_rate": 0.8,
  "error_rate": 0.2,
  "avg_latency_ms": 0.0,
  "human_review_total": 6,
  "human_review_rate": 0.6
}

2026-10-26 15:49:09,802 INFO endpoint: /score, status: 200, latency: 18.08ms, txn_id: T800001, prob: 0.0430
2026-10-26 15:49:09,843 WARNING endpoint: /score, status: 200, latency: 15.30ms, txn_id: T800003, prob: 0.1765


Limitations:
* The training dataset relies entirely on synthetic data combined with hand-built heuristic rules rather than real-world transaction distributions.
* The system utilizes a scripted stand-in mock function in place of a fully trained, dynamically fine-tuned language model.
* Information retrieval depends on a basic keyword-based matching approach that cannot effectively resolve semantic meaning or contextual intent.
* The classification decision threshold is tuned on a very small set of validation fraud cases, making it sensitive to distribution shifts in production.