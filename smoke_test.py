import time
import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"

# Payload templates as defined in Part D
BENIGN = {
    "merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI", "amount_inr": 1500,
    "hour_of_day": 13, "txn_count_last_24h": 0, "avg_amount_30d_inr": 1600,
    "distance_from_home_km": 3.0, "is_new_device": 0, "account_age_days": 900
}

RISKY = {
    "merchant_category": "Electronics", "city": "Mumbai", "payment_mode": "Card", "amount_inr": 45000,
    "hour_of_day": 2, "txn_count_last_24h": 9, "avg_amount_30d_inr": 3000,
    "distance_from_home_km": 60.0, "is_new_device": 1, "account_age_days": 40
}

BIG = BENIGN.copy()
BIG.update({
    "merchant_category": "Utilities", "payment_mode": "NetBanking",
    "amount_inr": 150000, "avg_amount_30d_inr": 250000,
    "distance_from_home_km": 0.0, "is_new_device": 1, "account_age_days": 2200
})

def run_smoke_tests():
    print("Waiting for service to start...")
    # Wait up to 120 seconds for health check
    for _ in range(60):
        try:
            res = requests.get(f"{BASE_URL}/health")
            if res.status_code == 200:
                print("Service is up!")
                break
        except requests.ConnectionError:
            time.sleep(2)
    else:
        print("Service failed to start in time.")
        return

    
    # Define the 11 smoke test requests (Q1 to Q11)
    tests = [
        {"id": "Q1", "category": "happy", "method": "GET", "endpoint": "/health", "expected_status": 200},
        {"id": "Q2", "category": "happy", "method": "POST", "endpoint": "/score", "payload": {**BENIGN, "txn_id": "T800001"}, "expected_status": 200},
        {"id": "Q3", "category": "escalation", "method": "POST", "endpoint": "/score", "payload": {**RISKY, "txn_id": "T800002"}, "expected_status": 200},
        {"id": "Q4", "category": "escalation", "method": "POST", "endpoint": "/score", "payload": {**BIG, "txn_id": "T800003"}, "expected_status": 200},
        {"id": "Q5", "category": "edge", "method": "POST", "endpoint": "/score", "payload": {**BENIGN, "txn_id": "T800004", "amount_inr": -5}, "expected_status": 400},
        {"id": "Q6", "category": "edge", "method": "POST", "endpoint": "/score", "payload": {"txn_id": "T800005", "merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI", "amount_inr": 1500, "txn_count_last_24h": 0, "avg_amount_30d_inr": 1600, "is_new_device": 0, "account_age_days": 900}, "expected_status": 422},
        {"id": "Q7", "category": "escalation", "method": "POST", "endpoint": "/case", "payload": {**RISKY, "txn_id": "T900002", "question": "more than 6 UPI payments in 24 hours velocity alert"}, "expected_status": 200},
        {"id": "Q8", "category": "escalation", "method": "POST", "endpoint": "/case", "payload": {**RISKY, "txn_id": "T900005", "question": "electronics merchant payment on an account younger than 30 days"}, "expected_status": 200},
        {"id": "Q9", "category": "edge", "method": "POST", "endpoint": "/case", "payload": {**RISKY, "txn_id": "T900099", "question": "card payment above INR 50000 from a new device"}, "expected_status": 503},
        {"id": "Q10", "category": "happy", "method": "POST", "endpoint": "/case", "payload": {**BENIGN, "txn_id": "T800001", "question": "card payment above INR 50000 from a new device"}, "expected_status": 200},
        {"id": "Q11", "category": "edge", "method": "POST", "endpoint": "/case", "payload": {**RISKY, "txn_id": "T800006", "question": "what is the cafeteria menu today"}, "expected_status": 200},
    ]

    pass_counts = {"happy": 0, "escalation": 0, "edge": 0}
    total_passes = 0

    for t in tests:
        url = f"{BASE_URL}{t['endpoint']}"
        try:
            if t["method"] == "GET":
                resp = requests.get(url)
            else:
                resp = requests.post(url, json=t.get("payload"))
            
            status_match = (resp.status_code == t["expected_status"])
            if status_match:
                print(f"[{t['id']}] {t['category'].upper()} - PASS (Status: {resp.status_code})")
                pass_counts[t['category']] += 1
                total_passes += 1
            else:
                print(f"[{t['id']}] {t['category'].upper()} - FAIL (Expected {t['expected_status']}, got {resp.status_code})")
        except Exception as e:
            print(f"[{t['id']}] ERROR: {e}")

    print("\n--- Per-Category Pass Counts ---")
    for cat, count in pass_counts.items():
        print(f"{cat}: {count} passes")
    
    print(f"\nTotal Passes: {total_passes} / {len(tests)}")

    # Fetch and display metrics JSON
    try:
        metrics_resp = requests.get(f"{BASE_URL}/metrics")
        print("\n--- Metrics JSON Response ---")
        print(json.dumps(metrics_resp.json(), indent=2))
    except Exception as e:
        print("Could not fetch metrics:", e)

if __name__ == "__main__":
    run_smoke_tests()