import base64
import requests
import json

# === Configuration ===
url = "http://localhost:5000/search"
image_path = "https://media.istockphoto.com/id/1019835828/photo/mango-and-leaf-isolated-white-background.jpg?s=612x612&w=0&k=20&c=_nmOBzO9mGEitT2rUvO1xAX9jwL5mHYI8AFRbYeyy-A="  # <-- Replace this with your image file path



# === Construct payload ===
payload = {
    "stream": False,
    "messages": [
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "what do we see in the image?" },
                {
                    "type": "image_url",
                    "image_url": { "url": image_path }
                }
            ]
        }
    ],
    "stream" : True
}

# === Send POST request ===
response = requests.post(url, json=payload)

# === Print response ===
print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except:
    print("Response Text:", response.text)
