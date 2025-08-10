import asyncio
import os
import random
import shutil
import stat
import json
from urllib.parse import quote
from playwright.async_api import async_playwright

# Config
MAX_LINKS_TO_TAKE = 10
isHeadless = True

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    try:
        func(path)
    except Exception as e:
        print(f"[!] Failed to delete: {path} → {e}")

# --------------------------------------------
# Yahoo Search - Text
# --------------------------------------------
class YahooSearchAgentText:
    def __init__(self, custom_port=None):
        self.playwright = None
        self.context = None
        self.query_count = 0
        self.custom_port = custom_port or random.randint(9000, 9999)
        print(f"[INFO] YahooSearchAgent ready on port {self.custom_port}.")

    async def start(self):
        self.playwright = await async_playwright().start()
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=f"/tmp/chrome-user-data-{self.custom_port}",
            headless=isHeadless,
            args=[
                f"--remote-debugging-port={self.custom_port}",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--no-first-run",
                "--disable-default-apps",
                "--disable-sync",
                "--no-sandbox",
            ],
            user_agent=get_random_user_agent()
        )
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

    async def search(self, query, max_links=MAX_LINKS_TO_TAKE):
        blacklist = [
            "yahoo.com/preferences",
            "yahoo.com/account",
            "login.yahoo.com",
        ]
        results = []
        try:
            self.query_count += 1
            page = await self.context.new_page()
            search_url = f"https://search.yahoo.com/search?p={quote(query)}"
            await page.goto(search_url, timeout=20000)
            await page.wait_for_selector("h3.title a", timeout=15000)

            link_elements = await page.query_selector_all("h3.title a")
            for link in link_elements:
                if len(results) >= max_links:
                    break
                href = await link.get_attribute("href")
                if href and href.startswith("http") and not any(b in href for b in blacklist):
                    results.append(href)

            await page.close()
            print(f"[INFO] Yahoo search returned {len(results)} results for '{query}'")
        except Exception as e:
            print(f"❌ Yahoo search failed: {e}")
        return results

    async def close(self):
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()
        shutil.rmtree(f"/tmp/chrome-user-data-{self.custom_port}", ignore_errors=True)
        print(f"[INFO] YahooSearchAgent on port {self.custom_port} closed.")

# --------------------------------------------
# Yahoo Search - Images
# --------------------------------------------
class YahooSearchAgentImage:
    def __init__(self, custom_port=None):
        self.playwright = None
        self.context = None
        self.save_dir = "downloaded_images"
        self.custom_port = custom_port or random.randint(9000, 9999)

    async def start(self):
        self.playwright = await async_playwright().start()
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=f"/tmp/chrome-user-data-{self.custom_port}",
            headless=isHeadless,
            args=[
                f"--remote-debugging-port={self.custom_port}",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--no-first-run",
                "--disable-default-apps",
                "--disable-sync",
                "--no-sandbox",
            ],
            user_agent=get_random_user_agent()
        )

    async def search_images(self, query, max_images=10):
        results = []
        os.makedirs(self.save_dir, exist_ok=True)
        try:
            page = await self.context.new_page()
            search_url = f"https://images.search.yahoo.com/search/images?p={quote(query)}"
            await page.goto(search_url, timeout=20000)
            await page.wait_for_selector("li img", timeout=15000)

            img_elements = await page.query_selector_all("li img")
            for img in img_elements[:max_images]:
                src = await img.get_attribute("data-src") or await img.get_attribute("src")
                if src and src.startswith("http"):
                    results.append(src)

            await page.close()
            return results
        except Exception as e:
            print(f"[ERROR] Yahoo image search failed: {e}")
            return []

    async def close(self):
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()
        shutil.rmtree(f"/tmp/chrome-user-data-{self.custom_port}", ignore_errors=True)

# --------------------------------------------
# Main search functions
# --------------------------------------------
async def web_search(query):
    yahoo_agent = YahooSearchAgentText(custom_port=random.randint(9000, 9999))
    try:
        await yahoo_agent.start()
        await asyncio.sleep(random.uniform(1, 3))
        results = await yahoo_agent.search(query, max_links=MAX_LINKS_TO_TAKE)
        return results
    finally:
        await yahoo_agent.close()

async def image_search(query, max_images=10):
    yahoo_agent = YahooSearchAgentImage(custom_port=random.randint(9000, 9999))
    try:
        await yahoo_agent.start()
        await asyncio.sleep(random.uniform(1, 3))
        results = await yahoo_agent.search_images(query, max_images)
        return results
    finally:
        await yahoo_agent.close()

# --------------------------------------------
# Run Example
# --------------------------------------------
if __name__ == "__main__":
    async def main():
        queries = ["ballet dancer"]
        for q in queries:
            print(f"\n{'='*50}")
            print(f"Searching Yahoo for: {q}")
            print('='*50)
            text_results = await web_search(q)
            print("Text results:", text_results)

            # img_results = await image_search(q, max_images=5)
            # print("Image results:", img_results)

    asyncio.run(main())
