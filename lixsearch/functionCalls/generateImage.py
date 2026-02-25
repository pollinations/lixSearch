import requests
import base64
import asyncio
import random
import os
from dotenv import load_dotenv
from urllib.parse import quote
import sys
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_image_from_prompt(prompt, session_id):

    api_url = f"https://gen.pollinations.ai/image/{quote(prompt)}?model=flux&height=512&width=512&seed={random.randint(0, 1000)}&quality=high&enhance=true"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TOKEN')}"
        }
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()

    with open("generated_image.jpg", "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    create_image_from_prompt("a beautiful landscape", "session123")