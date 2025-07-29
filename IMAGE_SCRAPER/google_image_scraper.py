import json
from urllib.parse import quote
from playwright.async_api import async_playwright
import asyncio

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


async def run():
    agent = GoogleImageSearchAgent()
    await agent.start()

    results = await agent.search_images("wooden trail in forest", max_images=10)
    for i, item in enumerate(results):
        print(f"{i+1}. Image URL: {item['image_url']}")
        print(f"   Source: {item['source_page']}\n")

    await agent.close()

if __name__ == "__main__":
    asyncio.run(run())