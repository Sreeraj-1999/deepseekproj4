import httpx
import json

url = "http://localhost:5004/chat/stream/"
payload = {"question": "What is the load of coffee percolator?", "imo": ""}

with httpx.stream("POST", url, json=payload, timeout=120) as response:
    for line in response.iter_lines():
        if line.startswith("data:"):
            data = json.loads(line[5:])
            print(f"[{data['type']}] {data.get('content', '')[:100]}")