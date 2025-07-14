import requests

query = "artificial intelligence"
url = f"http://51.15.192.16:5000/test"

response = requests.get(url)

print(response)
