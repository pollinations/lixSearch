import asyncio
from urllib.parse import quote
from playwright.async_api import async_playwright
import json
import base64
import requests

async def scrape_bing_images(page, query, max_images):
    await page.goto(f"https://www.bing.com/images/search?q={quote(query)}", timeout=60000)
    await page.wait_for_timeout(3000)

    for _ in range(3):
        await page.mouse.wheel(0, 2000)
        await page.wait_for_timeout(1000)

    images = await page.query_selector_all("img")
    results = []
    for img in images[:max_images]:
        try:
            src = await img.get_attribute("src")
            if src and src.startswith("http"):
                results.append({"image_url": src, "source_page": page.url})
        except:
            continue
    return results

class GoogleImageSearchAgent:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.query_count = 0
        print("[INFO] GoogleImageSearchAgent initialized.")

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        self.context = await self._new_context()

    async def _new_context(self):
        context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/114.0.0.0 Safari/537.36",
            viewport={'width': 1280, 'height': 800},
            java_script_enabled=True,
            locale="en-US"
        )
        await context.add_init_script(
            """Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"""
        )
        return context

    async def search_images(self, query, max_images=30):
        try:
            if self.query_count >= 50:
                print("[INFO] Resetting browser context.")
                await self.context.close()
                self.context = await self._new_context()
                self.query_count = 0

            self.query_count += 1
            results = []

            search_url = f"https://www.google.com/search?tbm=isch&q={quote(query)}"
            page = await self.context.new_page()
            await page.goto(search_url, timeout=60000)
            await page.wait_for_selector("img", state="attached", timeout=10000)

            for _ in range(3):  # Scroll to load more
                await page.mouse.wheel(0, 2000)
                await page.wait_for_timeout(1000)

            images = await page.query_selector_all("img")
            for img in images:
                src = await img.get_attribute("src")
                if src and src.startswith("http") and not src.startswith("data:"):
                    results.append({
                        "image_url": src,
                        "source_page": search_url
                    })
                    if len(results) >= max_images:
                        break

            await page.close()
            return results
        except Exception as e:
            print("❌ Google image search failed:", e)
            return []

    async def close(self):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("[INFO] GoogleImageSearchAgent closed.")

async def scrape_all_images(query, max_images=20):
    all_results = {"bing": [], "google": [], "duckduckgo": []}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        print("Scraping Bing...")
        all_results["bing"] = await scrape_bing_images(page, query, max_images)

        print("Scraping Google...")
        google_agent = GoogleImageSearchAgent()
        google_agent.playwright = p
        google_agent.browser = browser
        google_agent.context = context
        all_results["google"] = await google_agent.search_images(query, max_images)

        await browser.close()
    return all_results


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def generate_image_query(image_path):
    image_base64 = image_to_base64(image_path)
    url = "https://text.pollinations.ai/openai"
    headers = {
        "Content-Type": "application/json"
    }

    detailed_instruction = (
        "Describe the contents of this image in the form of a natural, short human-like search query. "
        "Include as much detail as possible: setting, objects, people, text, signs, brands, logos, style "
        "would type into a search engine to find similar images. Prioritize clarity, descriptive precision, and keywords "
        "likely to improve search relevance."
        "Keep the prompt short and relevant, no need for huge long descriptions, just focus on the key elements that would help in searching for similar images."
    )

    data = {
        "model": "openai",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": detailed_instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "token": "fEWo70t94146ZYgk",
        "max_tokens": 100
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()

if __name__ == "__main__":
    query = "wooden boardwalk trail in marsh with trees"
    all_image_results = asyncio.run(scrape_all_images(query, max_images=20))

    for engine, images in all_image_results.items():
        print(f"\n--- {engine.upper()} RESULTS ---")
        for i, img in enumerate(images):
            print(f"{i+1}. Image URL: {img['image_url']}\n   Source Page: {img['source_page']}\n")
