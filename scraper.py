"""
Step 2: Scrape IMDb 2024 movies (Movie Name + Storyline) using Selenium.
Output: imdb_2024_movies.csv
Run:    python scraper.py
"""
import re
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

URL = ("https://www.imdb.com/search/title/?title_type=feature"
       "&release_date=2024-01-01,2024-12-31")
MAX_MOVIES = 1000           # TEST with 100 first; change to 1000 once it works
OUTPUT_CSV = "imdb_2024_movies.csv"

# IMDb changes its layout sometimes, so we try several selectors
ITEM_SELECTORS = [
    "li.ipc-metadata-list-summary-item",
    "div.dli-parent",
    "[data-testid='dli-parent']",
]
TITLE_SELECTORS = ["h3.ipc-title__text", "h3"]
PLOT_SELECTORS = [
    "div.ipc-html-content-inner-div",
    "[data-testid='dli-plot-container']",
]


def start_browser():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--lang=en-US")
    # Make Chrome look like a normal browser (IMDb blocks obvious bots)
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    return driver


def get_items(driver):
    for css in ITEM_SELECTORS:
        items = driver.find_elements(By.CSS_SELECTOR, css)
        if items:
            return items
    return []


def wait_for_list(driver):
    """Wait for the movie list. If IMDb shows a captcha, let the user solve it."""
    try:
        WebDriverWait(driver, 30).until(lambda d: len(get_items(d)) > 0)
        return
    except TimeoutException:
        driver.save_screenshot("debug_page.png")
        print("\nMovie list not found. Screenshot saved as debug_page.png")
        print("Check the Chrome window:")
        print(" - If you see a captcha / 'verify you are human', solve it.")
        print(" - If a cookie popup is shown, accept it.")
        input("Once you can see the movie list in Chrome, press Enter here... ")

    if not get_items(driver):
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        raise RuntimeError(
            "Still no movie list. Send debug_page.png to Claude, "
            "or use the GUVI dataset instead.")


def load_all_movies(driver):
    """Keep clicking '50 more' until MAX_MOVIES are loaded or no button left."""
    while True:
        count = len(get_items(driver))
        print(f"Loaded {count} movies...")
        if count >= MAX_MOVIES:
            break
        buttons = driver.find_elements(By.CSS_SELECTOR, "button.ipc-see-more__button")
        if not buttons:
            print("No more '50 more' button - all movies loaded.")
            break
        btn = buttons[0]
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
        time.sleep(1)
        driver.execute_script("arguments[0].click();", btn)
        try:
            WebDriverWait(driver, 20).until(lambda d: len(get_items(d)) > count)
        except TimeoutException:
            print("Page stopped loading new movies - stopping here.")
            break


EXTRACT_JS = """
const items = document.querySelectorAll(arguments[0]);
const titleCss = arguments[1], plotCss = arguments[2];
return Array.from(items).map(li => {
  // 1) Try the known IMDb class names
  const t = li.querySelector(titleCss);
  const p = li.querySelector(plotCss);
  let title = t ? t.textContent.trim() : '';
  let plot  = p ? p.textContent.trim() : '';

  // 2) Fallback: don't depend on class names at all
  const texts = Array.from(li.querySelectorAll('*'))
      .map(e => (e.textContent || '').trim())
      .filter(x => x.length > 0);
  if (!/^\\d+\\.\\s+\\S/.test(title)) {
      // title looks like "200. The Shrouds" -> shortest text matching that
      const m = texts.filter(x => /^\\d+\\.\\s+\\S/.test(x))
                     .sort((a, b) => a.length - b.length);
      title = m.length ? m[0] : title;
  }
  if (plot.length < 30) {
      // storyline = longest text from an element with no child elements
      const leaves = Array.from(li.querySelectorAll('*'))
          .filter(e => e.children.length === 0)
          .map(e => (e.textContent || '').trim())
          .filter(x => x !== title)
          .sort((a, b) => b.length - a.length);
      plot = (leaves.length && leaves[0].length >= 30) ? leaves[0] : '';
  }
  return [title, plot];
});
"""


def extract_movies(driver):
    """Read all titles + storylines in one JavaScript call.
    (textContent works even for movies that are off-screen,
     unlike Selenium's .text which returns empty for them.)"""
    rows = []
    for css in ITEM_SELECTORS:
        rows = driver.execute_script(
            EXTRACT_JS, css, ", ".join(TITLE_SELECTORS), ", ".join(PLOT_SELECTORS))
        if rows:
            # save one movie's HTML so we can debug if something goes wrong
            html = driver.execute_script(
                "const e=document.querySelector(arguments[0]);"
                "return e ? e.outerHTML : '';", css)
            with open("debug_item.html", "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Selector used: {css}  |  items found: {len(rows)}")
            print("Sample rows:", rows[:3])
            break

    movies = []
    for title, story in rows[:MAX_MOVIES]:
        if not title:
            continue
        title = re.sub(r"^\d+\.\s*", "", title)        # remove "1. " numbering
        movies.append({"Movie Name": title, "Storyline": story})
    print(f"Extracted {len(movies)} movies from the page.")
    return movies


def main():
    driver = start_browser()
    try:
        driver.get(URL)
        wait_for_list(driver)
        load_all_movies(driver)
        movies = extract_movies(driver)
    finally:
        driver.quit()

    df = pd.DataFrame(movies)
    if df.empty:
        print("No movies extracted. Use the GUVI dataset instead.")
        return
    df = df[df["Storyline"].str.len() > 0]                 # drop empty storylines
    df = df.drop_duplicates(subset="Movie Name").reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nDone! Saved {len(df)} movies to {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()