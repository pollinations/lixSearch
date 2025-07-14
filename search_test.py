import requests

query = "artificial intelligence"
url = f"https://search.pollinations.ai/test"

response = requests.get(url)

print(response.text)
