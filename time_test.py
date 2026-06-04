# time_test.py
import requests, time

def one_call():
    payload = {
        "model": "llama3.2:latest",
        "messages": [{"role": "user", "content": "Say the word OK and nothing else."}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 10}
    }
    t0 = time.time()
    r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=60)
    elapsed = time.time() - t0
    return elapsed, r.json()["message"]["content"]

print("Testing Ollama speed...")
for i in range(3):
    elapsed, response = one_call()
    print(f"Call {i+1}: {elapsed:.1f}s — '{response.strip()}'")

print(f"\nEstimated time for 1 full trial (3 calls): ~{elapsed*3:.0f}s")
print(f"Estimated time for 20 trials (60 calls): ~{elapsed*60/60:.0f} minutes")