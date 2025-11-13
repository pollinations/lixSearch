import requests
import json
import os 
from dotenv import load_dotenv
load_dotenv()

POLLINATIONS_ENDPOINT = "https://enter.pollinations.ai/api/generate/openai"
POLLINATIONS_TOKEN = os.getenv("TOKEN")

def send_message(system_prompt, user_message):
    payload = {
        "model": "openai",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {POLLINATIONS_TOKEN}"
    }
    
    response = requests.post(POLLINATIONS_ENDPOINT, json=payload, headers=headers)
    return response.json()

# Example usage
if __name__ == "__main__":
    system = "You are a helpful assistant."
    user = "What is Python?"
    result = send_message(system, user)
    print(json.dumps(result, indent=2))