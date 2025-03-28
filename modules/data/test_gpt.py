from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import json
import re
import json
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from bs4 import BeautifulSoup
import time
from pydantic import BaseModel
from typing import List, Dict
import torch
import time 


# Set up undetected ChromeDriver in headless mode
options = uc.ChromeOptions()
# options.add_argument('--headless')  # Run in headless mode
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = uc.Chrome(options=options)

# Load metadata from JSON file
with open("./metadata.json", 'r') as f:
    metadata = json.load(f)

# Get base URLs from metadata
base_url = metadata["Khoury College of Computer Science"]["Research_spaces"]["base_url"]
labs_url = metadata["Khoury College of Computer Science"]["Labs_groups"]["base_url"]


device = torch.device("mps")
# ✅ Load a free open-source model (LLaMA 3 or Mistral 7B)
extractor = pipeline("text-generation", model="microsoft/phi-2", device = device)

# ✅ Function to extract webpage content
# Function to extract page content in order
def extract_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "span"])
    content = [tag.get_text(separator=" ", strip=True) for tag in tags]
    return "\n".join(content)

# Function to extract lab information from text
def extract_lab_info(text):
    system_prompt = """
    Extract the following details from the given webpage content and return structured JSON format.
    Ensure the output is a valid JSON object and nothing else.

    {
      "about": "Short summary of the lab",
      "researches": [
        {
          "title": "Research project title",
          "abstract": "Brief description of the project"
        }
      ]
    }
    """

    response = extractor(f"{system_prompt}\n\n{text}", max_length=600, truncation=True)
    generated_text = response[0]["generated_text"]

    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("❌ Error: Generated JSON is invalid.")
            return {}
    else:
        print("❌ No valid JSON detected in the model output.")
        return {}

# Function to split content into blocks of 400 words
def split_text(text, max_words=400):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Main function to get lab groups
def get_labs_groups():
    # Initialize WebDriver (ensure ChromeDriver is installed)
    driver.get(labs_url)

    lab_elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.wp-block-khoury-link-list-item a'))
    )
    lab_urls = [lab.get_attribute("href") for lab in lab_elements]

    labs_data = {}

    for url in lab_urls:
        print(f"Processing: {url}")

        try:
            st_time = time.time()
            page_content = extract_page_content(url)

            if len(page_content.split()) > 400:
                chunks = split_text(page_content)
                structured_info = {"about": "", "researches": []}

                for chunk in chunks:
                    chunk_info = extract_lab_info(chunk)
                    structured_info["about"] += " " + chunk_info.get("about", "")
                    structured_info["researches"].extend(chunk_info.get("researches", []))
            else:
                structured_info = extract_lab_info(page_content)
            if labs_data.get(url,0) == 0:
                labs_data[url] = [structured_info]
            else:
                labs_data[url].append(structured_info)
            print(structured_info)

        except Exception as e:
            print(f"❌ Error processing {url}: {e}")

    # Save structured output as JSON
    with open("extracted_labs.json", "w") as f:
        json.dump(labs_data, f, indent=4)

    # Close Selenium driver
    driver.quit()
    
    
if __name__ == "__main__":
    print(get_labs_groups())