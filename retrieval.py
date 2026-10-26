"""retrieval.py - Policy Retrieval and Refusal Calibration."""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load policies
with open("fraud_policies.json", "r") as f:
    policies = json.load(f)

# Extract texts and IDs
policy_texts = [p["text"] for p in policies]
policy_ids = [p["policy_id"] for p in policies]

# Fit TfidfVectorizer with default settings (no stop-word removal)
vectorizer = TfidfVectorizer()
policy_matrix = vectorizer.fit_transform(policy_texts)

# Define MIN_SCORE (chosen between out-of-scope max 0.4251 and in-scope min 0.4862)
MIN_SCORE = 0.485

def retrieve(query):
    """
    Given a query string, returns (policy_id, policy_text, score) 
    for the policy with the highest cosine similarity.
    """
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, policy_matrix).flatten()
    best_idx = similarities.argmax()
    best_score = float(similarities[best_idx])
    
    return policy_ids[best_idx], policy_texts[best_idx], best_score

if __name__ == "__main__":
    with open("retrieval_queries.json", "r") as f:
        queries = json.load(f)
    
    print(f"{'Query':<30} | {'Expected':<10} | {'Retrieved':<10} | {'Score':<8}")
    print("-" * 68)
    
    hit_count = 0
    for item in queries:
        q_text = item["query"]
        expected = item.get("expected_policy")
        pid, ptext, score = retrieve(q_text)
        
        if expected and pid == expected:
            hit_count += 1
            
        print(f"{q_text[:28]:<30} | {str(expected):<10} | {pid:<10} | {score:.4f}")
        
    print(f"\nIn-scope top-1 hit count: {hit_count}")