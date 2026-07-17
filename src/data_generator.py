import json
import os
from langchain_ollama import ChatOllama

def generate_transactions():
    # Enforcing JSON output directly at the model level
    llm = ChatOllama(model="llama3.2", temperature=0.7, format="json")
    
    prompt = """
    You are an expert synthetic data generator for an Anti-Money Laundering (AML) system.
    Generate a JSON array containing exactly 20 realistic bank transactions.
    
    Requirements:
    - Include exactly 2 instances of 'Smurfing' (multiple transfers between 9,500 and 9,990 EUR).
    - Include exactly 1 instance of 'Velocity Fraud' (3 or more rapid transfers between the same accounts).
    - The remaining 17 transactions should be normal everyday spending (groceries, rent, salary).
    
    Output strictly a JSON array of objects. Each object must have these exact keys:
    - "transaction_id" (string)
    - "sender_account" (string)
    - "receiver_account" (string)
    - "amount" (float)
    - "date" (string, ISO 8601 format)
    - "currency" (always "EUR")
    - "is_fraudulent" (boolean)
    - "fraud_type" (string: "Smurfing", "Velocity Fraud", or "None")
    """
    
    print("Prompting Llama-3.2 to generate a batch of 20 transactions...")
    response = llm.invoke(prompt)
    
    try:
        data = json.loads(response.content.strip())
        
        os.makedirs("data", exist_ok=True)
        with open("data/transactions.jsonl", "a", encoding="utf-8") as f:
            for tx in data:
                f.write(json.dumps(tx) + "\n")
        print(f"Successfully appended {len(data)} transactions to data/transactions.jsonl")
        
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw output:")
        print(response.content)

if __name__ == "__main__":
    # Loop 5 times to quickly generate 100 transactions to start testing with
    for i in range(5):
        print(f"\n--- Generating Batch {i+1}/5 ---")
        generate_transactions()