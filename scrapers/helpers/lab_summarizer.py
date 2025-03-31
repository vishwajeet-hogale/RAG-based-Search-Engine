import base64
import csv
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
from tqdm import tqdm
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
# 1. Load the summarization model (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

def decode_html(encoded):
    return base64.b64decode(encoded).decode('utf-8')

# 2. Clean raw HTML
def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Extract <title>
    title = soup.title.string.strip() if soup.title else ""

    # Extract <meta name="description">
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()

    # Extract headings (h1–h3)
    headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]

    # Extract key paragraphs (limit to first 5)
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    paragraphs = paragraphs[0:20]

    # Combine everything
    parts = [title, meta_desc] + headings + paragraphs
    text = '\n'.join(part for part in parts if part)

    return text

# 3. Chunk cleaned text into ~500-word pieces
def chunk_text(text, max_words=500):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# 4. Summarize each chunk
def summarize_chunks(chunks, batch_size=4):
    summaries = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            input_tokens = [len(tokenizer.encode(text)) for text in batch]
            max_length = max(int(t * 0.8) for t in input_tokens)  # aim for ~70% compression
            min_length = max(30, int(max_length * 0.5))

            results = summarizer(
                batch,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            summaries.extend([r['summary_text'] for r in results])
        except Exception as e:
            print(f"Batch summarization error: {e}")
            summaries.extend([""] * len(batch))  # Fill in empty summaries if batch fails

    return summaries


# 5. Full summary: summarize the summaries
def summarize_full(text):
    chunks = chunk_text(text)
    chunk_summaries = summarize_chunks(chunks)
    combined = ' '.join(chunk_summaries)
    if combined.strip():
        try:
            words = combined.split()
            max_toks = min(500, int(len(words)))
            min_toks = min(30, max_toks // 2)
            final_summary = summarizer(combined, max_length=max_toks, min_length=min_toks, do_sample=False)[0]['summary_text']
            return final_summary
        except Exception as e:
            print(f"Final summarization error: {e}")
            return ''
    else:
        return ''

# 6. Load CSV with HTML and process
def summarize_labs(csv_in='../modules/data/labs_with_html.csv', csv_out='../modules/data/labs_with_summaries.csv'):
    df = pd.read_csv(csv_in)
    summaries = []

    for html in tqdm(df['HTML'], desc="Summarizing lab pages"):
        try:
            decoded_html = decode_html(html)
            clean = clean_html(decoded_html)
            summary = summarize_full(clean)
        except Exception as e:
            print(f"Error processing row: {e}")
            summary = ''
        summaries.append(summary)

    df['Summary'] = summaries
    df_subset = df[["Research Area","Lab Name","Link","Summary"]]
    df_subset.to_csv(csv_out,
        index=False,
        encoding='utf-8',
        quoting=csv.QUOTE_ALL,
        escapechar='\\')
    print(f"Saved summarized labs to {csv_out}")

if __name__ == "__main__":
    summarize_labs()
