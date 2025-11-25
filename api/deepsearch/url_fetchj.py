

import asyncio
from playwright.async_api import async_playwright

async def get_transcript_url(video_url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--window-size=1920,1080",
                "--start-maximized",
                "--autoplay-policy=no-user-gesture-required"
            ]
        )

        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        page = await context.new_page()

        # 🥷 Stealth patch to remove headless indicators
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        """)

        transcript_url = None

        def capture_url(req):
            url = req.url
            if (
                "timedtext" in url or
                "texttrack" in url or
                "caption" in url
            ):
                nonlocal transcript_url
                transcript_url = url

        page.on("request", capture_url)

        await page.goto(video_url, wait_until="networkidle")

        # Force YouTube player UI visibility
        await page.evaluate("""
            const player = document.querySelector('.html5-video-player');
            if (player) player.classList.add('ytp-autohide');
        """)

        # Ensure CC button is interactable
        await page.wait_for_selector('button.ytp-subtitles-button.ytp-button', state="visible")

        # Click fake mouse movement (important in headless)
        await page.mouse.move(300, 300)
        await page.wait_for_timeout(300)

        await page.click('button.ytp-subtitles-button.ytp-button', force=True)

        # Give time for transcript request to fire
        await page.wait_for_timeout(2500)

        await browser.close()
        return transcript_url


# --------- TEST ----------
if __name__ == "__main__":
    url = asyncio.run(
        get_transcript_url("https://www.youtube.com/watch?v=pdNo8096vbc")
    )
    print("Transcript URL:", url)