"""casenote.py - Policy-Grounded Case Notes generation, validation, and execution."""
import json
from retrieval import retrieve, MIN_SCORE
import mock_llm  # Assuming mock_llm is provided in the environment

# JSON Schema definition for validation
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_team": {"type": "string", "enum": ["card_risk", "upi_risk", "merchant_ops", "customer_care"]},
        "recommended_action": {"type": "string", "enum": ["approve", "hold_for_review", "block_and_notify"]},
        "policy_id": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5", "P6"]}
    },
    "required": ["summary", "priority", "recommended_team", "recommended_action", "policy_id"]
}

def build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text):
    prompt = f"""Role: You are an expert fraud risk analyst assistant.
Task: Write a precise case note based strictly on the provided fraud policy text.
Constraints:
- Use only the provided policy text to determine recommendations.
- Reply with ONE valid JSON object and nothing else.

Output Format:
A JSON object with the exact keys: summary, priority, recommended_team, recommended_action, policy_id.

TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy_id}

Policy Text:
{policy_text}
"""
    return prompt

def validate(text, retrieved_policy_id):
    """
    Validates the LLM output through 3 gates:
    Gate 1: Valid JSON object.
    Gate 2: All 5 required keys present and are strings.
    Gate 3: Enums match allowed values and policy_id equals retrieved_policy_id.
    """
    # Gate 1: JSON parsing
    try:
        note = json.loads(text)
        if not isinstance(note, dict):
            return None, 1
    except (json.JSONDecodeError, TypeError):
        return None, 1

    # Gate 2: Required keys and types
    required_keys = SCHEMA["required"]
    for key in required_keys:
        if key not in note or not isinstance(note[key], str):
            return None, 2

    # Gate 3: Enum values and policy match
    if note["priority"] not in SCHEMA["properties"]["priority"]["enum"]:
        return None, 3
    if note["recommended_team"] not in SCHEMA["properties"]["recommended_team"]["enum"]:
        return None, 3
    if note["recommended_action"] not in SCHEMA["properties"]["recommended_action"]["enum"]:
        return None, 3
    if note["policy_id"] not in SCHEMA["properties"]["policy_id"]["enum"] or note["policy_id"] != retrieved_policy_id:
        return None, 3

    return note, 0

def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=3):
    pid, ptext, score = retrieve(question)
    
    # Check refusal threshold
    if score < MIN_SCORE:
        return {
            "status": "no_policy_match",
            "attempts": 0,
            "failures": [],
            "case_note": None,
            "safe_default": False,
            "guardrail_override": False
        }
        
    prompt = build_prompt(txn_id, fraud_probability, amount_inr, pid, ptext)
    
    attempts = 0
    failures = []
    case_note = None
    safe_default = False
    
    while attempts < max_attempts:
        attempts += 1
        # ConnectionError from mock_llm.generate will propagate naturally as required
        response_text = mock_llm.generate(prompt)
        
        note, gate = validate(response_text, pid)
        if gate == 0:
            case_note = note
            break
        else:
            failures.append(gate)
            
    # If all attempts failed, apply safe default
    if case_note is None:
        safe_default = True
        case_note = {
            "summary": "Automated case note unavailable; manual review required.",
            "priority": "high",
            "recommended_team": "customer_care",
            "recommended_action": "hold_for_review",
            "policy_id": pid
        }
        
    # Output guardrail
    guardrail_override = False
    if case_note["recommended_action"] == "approve" and fraud_probability >= 0.5:
        case_note["recommended_action"] = "hold_for_review"
        guardrail_override = True
        
    return {
        "status": "ok",
        "attempts": attempts,
        "failures": failures,
        "case_note": case_note,
        "safe_default": safe_default,
        "guardrail_override": guardrail_override
    }

if __name__ == "__main__":
    mock_llm.reset()
    with open("case_scenarios.json", "r") as f:
        scenarios = json.load(f)
        
    print(f"{'Scenario':<15} | {'Retrieved':<10} | {'Attempts':<8} | {'Failures':<10} | {'Status':<6} | {'Safe Def':<8} | {'Final Action':<16} | {'Override':<8}")
    print("-" * 95)
    
    for idx, s in enumerate(scenarios, start=1):
        # Support different possible key names for scenario identifier in JSON
        scenario_name = s.get("scenario") or s.get("id") or f"S{idx}"
        
        res = run_case(
            txn_id=s["txn_id"],
            fraud_probability=s["fraud_probability"],
            amount_inr=s["amount_inr"],
            question=s["question"]
        )
        
        retrieved_pid = res["case_note"]["policy_id"] if res["case_note"] else "N/A"
        final_action = res["case_note"]["recommended_action"] if res["case_note"] else "N/A"
        
        print(f"{str(scenario_name):<15} | {retrieved_pid:<10} | {res['attempts']:<8} | {str(res['failures']):<10} | {res['status']:<6} | {str(res['safe_default']):<8} | {final_action:<16} | {str(res['guardrail_override']):<8}")