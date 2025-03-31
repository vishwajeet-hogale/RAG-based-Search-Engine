import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import pipeline
from playwright.async_api import async_playwright
from html import unescape
import asyncio
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ✅ 1. Summarizer (BART on CUDA)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

# ✅ 2. Extract abstract from HTML
from html import unescape

from bs4 import BeautifulSoup
from html import unescape

from urllib.parse import urlparse
from bs4 import BeautifulSoup
from html import unescape

def extract_abstract(html, url=""):
    soup = BeautifulSoup(html, "html.parser")
    text = ""
    domain = urlparse(url).netloc.lower()

    # 🔹 Custom logic for NDSS Symposium
    if "ndss-symposium.org" in domain:
        paper_div = soup.find("div", class_="paper-data")
        if paper_div:
            paragraphs = paper_div.find_all("p")
            text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

    # 🔹 Try standard abstract containers
    elif not text:
        abstract_section = (
            soup.find("section", id="abstract") or
            soup.find("div", id="abstracts") or
            soup.find("section", id="abstracts") or
            soup.find("div", id="abstract")
        )

        if abstract_section:
            paragraphs = abstract_section.find_all(["p", "div"], attrs={"role": "paragraph"})
            if not paragraphs:
                text = abstract_section.get_text(separator=" ", strip=True)
            else:
                text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        else:
            block = soup.find("blockquote", class_="abstract")
            if block:
                text = block.get_text(strip=True).replace("Abstract:", "").strip()
            else:
                possible_abstracts = soup.find_all(lambda tag:
                    tag.name in ["p", "div", "section"] and (
                        any("abstract" in (cls or "").lower() for cls in (tag.get("class") or [])) or
                        "abstract" in (tag.get("id") or "").lower()
                    )
                )

                if not possible_abstracts:
                    possible_abstracts = soup.find_all("p", limit=5)

                text = " ".join(p.get_text(separator=" ", strip=True) for p in possible_abstracts)

    # ✅ Clean and normalize whitespace
    text = text.replace("\n", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")

    return unescape(text) if len(text) else "No summary"


# ✅ 3. Hierarchical summarization
def summarize_long_text(text, max_chunk_words=500):
    words = text.split()
    chunks = [' '.join(words[i:i + max_chunk_words]) for i in range(0, len(words), max_chunk_words)]
    chunk_summaries = []

    for idx, chunk in enumerate(chunks):
        try:
            max_len = min(300, int(len(chunk.split())))
            min_len = max(30, int(max_len * 0.5))
            summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
            chunk_summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Chunk {idx+1} summarization error: {e}")
            continue

    combined = ' '.join(chunk_summaries)
    if not combined.strip():
        return "Summarization failed"

    # 🔒 Check token count before final summarization
    final_tokens = tokenizer(combined, return_tensors="pt").input_ids.shape[1]
    if final_tokens > 1024:
        print(f"Combined summary too long ({final_tokens} tokens), chunking again...")

        # Rechunk and resummarize
        final_chunks = [' '.join(combined.split()[i:i + max_chunk_words]) for i in range(0, len(combined.split()), max_chunk_words)]
        final_summaries = []

        for idx, f_chunk in enumerate(final_chunks):
            try:
                max_len = min(300, int(len(f_chunk.split())))
                min_len = max(30, int(max_len * 0.5))
                summary = summarizer(f_chunk, max_length=max_len, min_length=min_len, do_sample=False)
                final_summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Final chunk {idx+1} error: {e}")
                continue

        return ' '.join(final_summaries).strip()

    # ✅ If token count is safe, do final single-shot summary
    try:
        combined_word_count = len(combined.split())
        max_len = min(300, int(combined_word_count))
        min_len = max(30, int(max_len * 0.5))

        final = summarizer(combined, max_length=max_len, min_length=min_len, do_sample=False)
        return final[0]['summary_text']
    except Exception as e:
        print(f"Final summary error: {e}")
        return "Final summarization failed"


# ✅ 4. Safe summarizer wrapper
def summarize_text(text):
    if not text or len(text.split()) < 300:
        return text  # Too short to summarize
    if len(text.split()) <= 500:
        return summarize_long_text(text, max_chunk_words=500)  # Single-shot if short
    return summarize_long_text(text)  # Hierarchical for long inputs

# ✅ 5. Async stealth browser to fetch HTML
async def fetch_html_playwright(url, wait=15):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
            java_script_enabled=True,
            locale="en-US"
        )
        page = await context.new_page()

        await page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        window.navigator.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
        """)

        try:
            await page.goto(url, timeout=60000)
            await page.wait_for_timeout(wait * 1000)
            html = await page.content()
            await browser.close()
            return html
        except Exception as e:
            print(f"Error loading {url}: {e}")
            await browser.close()
            return ""

# ✅ 6. Main async loop with checkpoint saving
async def process_publications(csv_in, csv_out, checkpoint_interval=100):
    df = pd.read_csv(csv_in)
    summaries = []

    for idx, link in enumerate(tqdm(df["Publication Link"], desc="Scraping & summarizing")):
        if not isinstance(link, str) or not link.startswith("http"):
            summaries.append("Invalid URL")
            continue

        if str(link).lower().endswith(".pdf") or "pdf" in str(link).lower():
            summaries.append("Skipped (PDF)")
        else:
            max_retries = 2
            retry_count = 0
            summary = ""

            while retry_count < max_retries:
                html = await fetch_html_playwright(link)
                if not html:
                    retry_count += 1
                    continue

                abstract = extract_abstract(html,link)

                # Check for CAPTCHA/Cloudflare text
                if "verify you are human" in abstract.lower():
                    print(f"CAPTCHA triggered on try {retry_count + 1} for: {link}")
                    retry_count += 1
                    await asyncio.sleep(2 * retry_count)  # backoff
                    continue

                summary = summarize_text(abstract)
                break  # exit loop if successful

            if not summary:
                summaries.append("Failed after retries")
                with open("captcha_failed_links.txt", "a") as f:
                    f.write(link + "\n")
            else:
                summaries.append(summary)

        # Save progress every 100 rows
        if (idx + 1) % checkpoint_interval == 0 or idx == len(df) - 1:
            df_partial = df.copy()
            df_partial["Publication Summary"] = summaries + [""] * (len(df_partial) - len(summaries))
            df_partial.to_csv(csv_out, index=False, encoding="utf-8", quoting=1)
            print(f"Saved checkpoint at {idx + 1} / {len(df)}")

    print(f"Final CSV saved to {csv_out}")

# ✅ Run the pipeline
if __name__ == "__main__":
    asyncio.run(process_publications(
        "failed_retries.csv",
        "failed_retries_fixed.csv"
    ))