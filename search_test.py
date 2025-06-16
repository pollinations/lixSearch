import requests

query = "artificial intelligence"
url = f"http://51.15.192.16:4000/search"

response = requests.get(url, params={
    "q": query,
    "format": "json"
})

data = response.json()
for result in data.get("results", []):
    print(result["title"], "-", result["url"])
