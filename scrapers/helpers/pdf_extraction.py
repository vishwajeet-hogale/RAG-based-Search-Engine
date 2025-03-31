# import pandas as pd
# import fitz  # PyMuPDF
# import re
# import os
# import requests
# from tqdm import tqdm

# # ========== CONFIG ==========
# INPUT_CSV = "skipped_pdfs.csv"
# OUTPUT_CSV = "pdfs_extracted_abstracts.csv"
# PDF_COLUMN = "Publication Link"
# ABSTRACT_COLUMN = "Publication Summary"
# TMP_FOLDER = "pdf_tmp"
# os.makedirs(TMP_FOLDER, exist_ok=True)

# # ========== ABSTRACT EXTRACTION ==========
# def extract_abstract_from_pdf(pdf_path):
#     try:
#         doc = fitz.open(pdf_path)
#         text = "\n".join([page.get_text() for page in doc])
#         text = re.sub(r'\s+', ' ', text)

#         # Look for abstract followed by content, stopping at intro or numbered section
#         match = re.search(
#             r'(abstract[\s:.-]*)(.*?)(?=\s*(introduction|1\s*\.\s|i\s*\.\s))',
#             text,
#             re.IGNORECASE
#         )
#         if match:
#             abstract = match.group(2).strip()
#             return abstract if len(abstract) > 30 else "Abstract too short"
#         else:
#             return "Abstract not found"
#     except Exception as e:
#         return f"Error reading PDF: {e}"

# # ========== DOWNLOAD PDF ==========
# def download_pdf(url, out_path):
#     try:
#         headers = {
#             "User-Agent": "Mozilla/5.0"
#         }
#         response = requests.get(url, headers=headers, timeout=15)
#         if response.status_code == 200 and response.headers.get("Content-Type", "").lower().startswith("application/pdf"):
#             with open(out_path, "wb") as f:
#                 f.write(response.content)
#             return True
#         return False
#     except Exception as e:
#         return False

# # ========== MAIN ==========
# def process_csv(input_csv, output_csv):
#     df = pd.read_csv(input_csv)
#     abstracts = []

#     for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing PDFs"):
#         url = str(row[PDF_COLUMN])
#         if not url.lower().endswith(".pdf") or not "pdf" in url.lower():
#             abstracts.append("Not a PDF")
#             continue

#         filename = os.path.join(TMP_FOLDER, f"doc_{i}.pdf")
#         success = download_pdf(url, filename)

#         if not success:
#             abstracts.append("Failed to download PDF")
#             continue

#         abstract = extract_abstract_from_pdf(filename)
#         abstracts.append(abstract)

#     df[ABSTRACT_COLUMN] = abstracts
#     df.to_csv(output_csv, index=False)
#     print(f"\n Saved to {output_csv}")

# # ========== RUN ==========
# if __name__ == "__main__":
#     process_csv(INPUT_CSV, OUTPUT_CSV)
import pandas as pd
import fitz  # PyMuPDF
import re
import os
import requests
from tqdm import tqdm

# ========== CONFIG ==========
INPUT_CSV = "skipped_pdfs.csv"
OUTPUT_CSV = "pdfs_extracted_abstracts.csv"
PDF_COLUMN = "Publication Link"
ABSTRACT_COLUMN = "Publication Summary"
TMP_FOLDER = "pdf_tmp"
os.makedirs(TMP_FOLDER, exist_ok=True)

# ========== TEXT EXTRACTORS ==========
def extract_section(text, start_keywords, stop_keywords, label="section"):
    start_pattern = r"(" + "|".join(re.escape(k) for k in start_keywords) + r")\s*[:.-]?\s*"
    stop_pattern = r"(?=\s*(" + "|".join(re.escape(k) for k in stop_keywords) + r"))"

    pattern = re.compile(start_pattern + r"(.*?)" + stop_pattern, re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        content = match.group(2).strip()
        return content if len(content) else f"{label} too short"
    return f"{label} not found"

def extract_sections_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        text = re.sub(r'\s+', ' ', text)

        abstract = extract_section(
            text,
            start_keywords=["abstract"],
            stop_keywords=["introduction", "1.", "i."],
            label="abstract"
        )

        conclusion = extract_section(
            text,
            start_keywords=["conclusion", "conclusions"],
            stop_keywords=["references", "acknowledgment", "acknowledgements", "bibliography"],
            label="conclusion"
        )

        return abstract, conclusion
    except Exception as e:
        return f"Error: {e}", f"Error: {e}"

# ========== DOWNLOAD ==========
def download_pdf(url, out_path):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
            with open(out_path, "wb") as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False

# ========== PROCESS ==========
def process_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    abstracts, conclusions = [], []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting from PDFs"):
        url = str(row[PDF_COLUMN])
        if not url.lower().endswith(".pdf") and not "pdf" in url.lower():
            abstracts.append("Not a PDF")
            conclusions.append("Not a PDF")
            continue

        filename = os.path.join(TMP_FOLDER, f"doc_{i}.pdf")
        success = download_pdf(url, filename)

        if not success:
            abstracts.append("Failed to download")
            conclusions.append("Failed to download")
            continue

        abstract, conclusion = extract_sections_from_pdf(filename)
        abstracts.append(abstract+conclusion)
        # conclusions.append(conclusion)

    # Add new columns
    df[ABSTRACT_COLUMN] = abstracts
    # df[CONCLUSION_COLUMN] = conclusions

    # Filter out failed rows
    clean_df = df[
        ~df[ABSTRACT_COLUMN].str.contains("not found|too short|Failed|Error|Not a", case=False) #&
        # ~df[CONCLUSION_COLUMN].str.contains("not found|too short|Failed|Error", case=False)
    ]

    # clean_df.to_csv(output_csv, index=False)
    df.to_csv("raw_pdfs.csv", index=False)
    print(f"\n✅ Clean data saved to {output_csv} with {len(clean_df)} rows.")

# ========== RUN ==========
if __name__ == "__main__":
    process_csv(INPUT_CSV, OUTPUT_CSV)
