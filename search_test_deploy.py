import requests
import json
import urllib.parse
url = "http://localhost:5000/search"
headers = {"Content-Type": "application/json"}
data = {
    "messages": [
        {"role": "user", "content": "what's 1+1?"}
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.text)
