import requests
from time import sleep
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from duckduckgo_search import DDGS
from config import MAX_LINKS_TO_TAKE

def google_search(query):
    blacklist = [
        "maps.google.", "support.google.", "accounts.google.", "policies.google.",
        "images.google.", "google.com/preferences", "https://www.instagram.com/reel",
        "https://www.instagram.com/p", "youtube.com/shorts", "youtube.com/live",
        "https://www.facebook.com/", "https://www.facebook.com/p"
    ]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=[
                "--disable-blink-features=AutomationControlled",
            ])
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/114.0.0.0 Safari/537.36",
                viewport={'width': 1280, 'height': 800},
                java_script_enabled=True,
                locale="en-US"
            )

            page = context.new_page()

            page.add_init_script(
                """Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"""
            )

            page.goto(f"https://www.google.com/search?q={query}", timeout=60000)
            page.wait_for_selector('a[jsname]')

            links = page.locator('a[jsname]')
            results = []

            for i in range(links.count()):
                href = links.nth(i).get_attribute("href")
                if href and href.startswith("http") and not any(bad in href for bad in blacklist):
                    results.append(href)

            browser.close()
            return results[:MAX_LINKS_TO_TAKE]
    except Exception as e:
        print("❌ Google search failed:", e)
        return []


def mojeek_form_search(query):
    url = "https://www.mojeek.com/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Referer": "https://www.mojeek.com/",
        "Accept": "text/html"
    }
    params = {"q": query}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.select("ul.results-standard li")
        links = []

        for r in results[:MAX_LINKS_TO_TAKE]:
            title_tag = r.select_one("a.title")
            if title_tag and title_tag.has_attr("href"):
                links.append(title_tag["href"])

        return links

    except requests.exceptions.RequestException as e:
        print("❌ Mojeek request failed:", e)
        return []

def ddgs_search(query):
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"
    }
    data = {"q": query}

    try:
        response = requests.post(url, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        links = []
        for h2 in soup.select("h2.result__title a[href^='http']"):
            href = h2.get("href")
            if (
                href
                and href.startswith("http")
                and not href.startswith("https://duckduckgo.com/y.js?")
            ):
                links.append(href)

        print(f"[INFO] DDG search completed with {len(links)} results.")
        return links[:MAX_LINKS_TO_TAKE]

    except Exception as e:
        print("❌ DDG search failed:", e)
        return []


def ddgs_search_module_search(query):
    results = []
    try:
        with DDGS() as ddgs:
            for entry in ddgs.text(query, max_results=MAX_LINKS_TO_TAKE):
                url = entry.get("href") or entry.get("link")
                if url and url.startswith("http"):
                    results.append(url)
        print(f"[INFO] DDG search returned {len(results)} links")
    except Exception as e:
        print("❌ DDG search failed:", e)
    return results

def web_search(query):
    print(f"[INFO] Running web search for: {query}")

    try:
        ddg_results = ddgs_search(query)
        if ddg_results:
            print(f"[INFO] Using DuckDuckGo HTML with {len(ddg_results)} results.")
            return ddg_results
        else:
            print("[INFO] DuckDuckGo HTML returned no results. Falling back to DDG module.")
    except Exception as e:
        print(f"[WARN] DuckDuckGo HTML search failed with error: {e}. Falling back to DDG module.")

    try:
        ddg_module_results = ddgs_search_module_search(query)
        if ddg_module_results:
            print(f"[INFO] Using DuckDuckGo module with {len(ddg_module_results)} results.")
            return ddg_module_results
        else:
            print("[INFO] DuckDuckGo module returned no results. Falling back to Mojeek.")
    except Exception as e:
        print(f"[WARN] DuckDuckGo module search failed with error: {e}. Falling back to Mojeek.")

    try:
        mojeek_results = mojeek_form_search(query)
        if mojeek_results:
            print(f"[INFO] Using Mojeek with {len(mojeek_results)} results.")
            return mojeek_results
        else:
            print("[INFO] Mojeek returned no results. Falling back to Google.")
    except Exception as e:
        print(f"[WARN] Mojeek search failed with error: {e}. Falling back to Google.")

    try:
        google_results = google_search(query)
        if google_results:
            print(f"[INFO] Using Google with {len(google_results)} results.")
            return google_results
        else:
            print("[INFO] Google returned no results.")
    except Exception as e:
        print(f"[WARN] Google search failed with error: {e}.")

    print("[INFO] All search engines failed to return results.")
    return []


if __name__ == "__main__":
    print("\nDDGS:")
    for url in ddgs_search("the best sign language detection using cnn"):
        print(url)

        
    # print("\nMOJEEK:")
    # for urls in mojeek_form_search("sign language detection using cnn"):
    #     print(urls)

    # print("\nGOOGLE:")
    # for url in google_search("jis university"):
    #     print(url)
    
    # print("\nWeb Search")
    # query = "Tell me something about quantum computing"
    # results = web_search(query)
    # print("Search results:")
    # for link in results:
    #     print(link)

