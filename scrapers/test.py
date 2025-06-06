import base64
import csv
import json
import re
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import requests
import pandas as pd
from scrapers.helpers.lab_summarizer import summarize_labs
from scrapers.helpers.paper_summarizer import process_publications
import urllib3
import torch
from itertools import islice


print(torch.cuda.is_available())           # Should be True
print(torch.cuda.get_device_name(0)) 

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import pandas as pd

# Set up undetected ChromeDriver in headless mode
options = uc.ChromeOptions()
options.add_argument('--headless')  # Run in headless mode
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = uc.Chrome(options=options)

# Load metadata from JSON file
with open("./metadata.json", 'r') as f:
    metadata = json.load(f)

# Get base URLs from metadata
base_url = metadata["Khoury College of Computer Science"]["Research_landing"]["base_url"]
research_url = metadata["Khoury College of Computer Science"]["Research_areas"]["base_url"]
institutes_and_centers_url = metadata["Khoury College of Computer Science"]["Institutes_and_centers"]["base_url"]
research_spaces_url = metadata["Khoury College of Computer Science"]["Research_spaces"]["base_url"]
labs_url = metadata["Khoury College of Computer Science"]["Labs_groups"]["base_url"]

# Initialize Chrome driver
options = uc.ChromeOptions()
options.headless = True  # Run in headless mode
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = uc.Chrome(options=options)

# Store the results
data = {}

# Function to get all research area URLs (Using Selenium if JavaScript is required)
def get_research_areas():
    print(f"Getting research areas")
    
    driver.get(base_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "main")))
    time.sleep(2)  # Allow JS to load

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    research_area_urls = {}
    research_areas = soup.find_all('li', class_='wp-block-khoury-link-list-item')
    
    for research in research_areas:
        area_name = research.text.strip()
        formatted_text = area_name.lower().replace(' ', '-')
        new_url = research_url + formatted_text
        research_area_urls[area_name] = new_url

    df = pd.DataFrame(list(research_area_urls.items()), columns=['Area', 'URL'])
    df.to_csv('../modules/data/research_areas.csv', index=False)
    return research_area_urls

# Function to get institutes and centers (Using Selenium)
def get_institutes_and_centers():
    print(f"Getting institutes and centers")
    research_data = []
    
    driver.get(institutes_and_centers_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "main")))
    time.sleep(2)  # Allow JS to load
    
    soup = BeautifulSoup(driver.page_source, "html.parser")

    institute_names = soup.select("main > div > div")

    if not institute_names:
        print("No institute names found")

    for name_tag in institute_names[1:]:
        name = name_tag.find("h3").text.strip()
        description_tag = name_tag.find("p")  # Get the paragraph next to h3
        a_element = name_tag.find("a")
        href = a_element['href'] if a_element and a_element.has_attr('href') else ""
        institute_data = {
            "name": name,
            "description": description_tag.text.strip() if description_tag else "",
            "href": href
        }
        research_data.append(institute_data)
    
    df = pd.DataFrame(research_data)
    df.to_csv("../modules/data/institutes_and_centers.csv", index=False)
    return research_data

# Function to get professor details (Using Selenium for expandable sections)
def get_professor_details(prof_url):
    """Fetch and parse a professor's page to extract details from collapsible sections."""
    print(f'Fetching professor details: {prof_url}')
    
    try:
        driver.get(prof_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "single-people__content")))
        time.sleep(2)  # Allow JS to load
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        content_sections = soup.select(".single-people__content div div div div div div")

        prof_data = {"sections": {}}

        for section in content_sections:
            h2_tag = section.find("h2")
            if not h2_tag:
                continue  # Skip if no header found

            section_name = h2_tag.text.strip()

            if section_name in ["Research interests", "Education"]:
                items = [li.text.strip() for li in section.select("ul li")]
                prof_data["sections"][section_name] = items

            elif section_name == "Biography":
                bio_text = section.find("p").text.strip() if section.find("p") else ""
                prof_data["sections"][section_name] = bio_text

            elif section_name == "Recent publications":
                publications = []
                for pub in section.select("ul li"):
                    time_tag = pub.find("time", class_="text-card__date")
                    date = time_tag.text.strip() if time_tag else ""

                    a_tag = pub.find("a")
                    pub_name = a_tag.text.strip() if a_tag else "Unknown Publication"
                    pub_link = a_tag["href"] if a_tag and a_tag.has_attr("href") else ""

                    citation_div = pub.find("div", class_="text-card__citation")
                    citation_text = citation_div.text.strip().split("Citation:")[1] if citation_div else ""

                    publications.append({"date": date, "publication": pub_name, "link": pub_link, "citation": citation_text})

                prof_data["sections"][section_name] = publications

        return prof_data

    except Exception as e:
        print(f"Error fetching professor details: {e}")
        return {}

def save_publications_per_row(data, filename="../modules/data/professor_details.csv"):
    rows = []

    for area, profs in data.items():
        for prof in profs:
            name = prof.get("name", "")
            url = prof.get("url", "")
            sections = prof.get("info", {}).get("sections", {})

            biography = sections.get("Biography", "")
            interests = "; ".join(sections.get("Research interests", []))
            education = "; ".join(sections.get("Education", []))
            publications = sections.get("Recent publications", [])

            # Create one row per publication
            for pub in publications:
                row = {
                    "Research Area": area,
                    "Professor Name": name,
                    "Profile URL": url,
                    "Biography": biography,
                    "Research Interests": interests,
                    "Education": education,
                    "Publication Date": pub.get("date", "").split("Published: ")[1],
                    "Publication Title": pub.get("publication", ""),
                    "Publication Link": pub.get("link", ""),
                    "Citation": pub.get("citation", "")
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.drop_duplicates(inplace=True)
    df.to_csv(filename, index=False)
    # print(f"Saved {len(rows)} publication records to {filename}")

# Function to get all research spaces URLs
def get_research_spaces():
    print(f"Getting research spaces")
    # Store the results
    data = {}
    research_spaces_df = []
    driver.get(research_spaces_url)
    try:
        # Using CSS Selector to handle class names with spaces
        research_spaces = WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.wp-block-column.is-layout-flow.wp-block-column-is-layout-flow'))
        )
        
        print(f"Found {len(research_spaces)} research spaces.")

        for research in research_spaces:
            try:
                title = research.find_element(By.TAG_NAME, 'h3').text
                description = research.find_element(By.TAG_NAME, 'p').text

                # Find the nested div that contains the <a> tag
                link_element = research.find_element(By.TAG_NAME, 'a')
                link = link_element.get_attribute('href')

                research_spaces_df.append([title, description, link])
            except Exception as e:
                print(f"Error extracting a research space: {e}")

    except Exception as e:
        print('Timeout while collecting research area URLs:', e)
    
    return pd.DataFrame(research_spaces_df, columns=["Lab", "Description", "Link"])

# Function to get professors by research area
def get_professors_by_area(area_name, area_url):
    print(f'Navigating to: {area_url}')
    response = requests.get(area_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    profs_list = []
    profs = soup.select("section ul li article div h3 a")

    for prof in profs:
        prof_name = prof.text.strip()
        prof_url = prof['href']
        prof_data = get_professor_details(prof_url)
        profs_list.append({"name": prof_name, "url": prof_url, "info":prof_data})
    
    return profs_list

def get_current_research_highlights():
    print(f"Getting research highlights")
    
    driver.get(base_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "main")))
    time.sleep(2)  # Allow JS to load
    
    soup = BeautifulSoup(driver.page_source, "html.parser")

    highlight_divs = soup.select("main > div > div:nth-of-type(6) > div:nth-of-type(2) > div > div > div > div")

    data = []

    for div in highlight_divs:
        try:
            h2_element = div.find("h2")
            h2_text = h2_element.text.strip() if h2_element else ""
        except:
            h2_text = ""

        try:
            p_element = div.find("p")
            p_text = p_element.text.strip() if p_element else ""
        except:
            p_text = ""

        try:
            link_element = div.find("a")
            href = link_element["href"] if link_element and link_element.has_attr("href") else ""
        except:
            href = ""

        data.append({
            "title": h2_text,
            "description": p_text,
            "link": href
        })

    df = pd.DataFrame(data)
    df.to_csv("../modules/data/current_research_highlights.csv", index=False)
    return data

def get_labs_links():
    print(f"Getting labs")
    lab_links = []
    names = []
    areas = []
    html = []

    for area, url in data['research_areas'].items():
    # for area, url in islice(data['research_areas'].items(), 2):
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            items = soup.select('.wp-block-khoury-link-list-item a')

            for tag in items:
                href = tag.get('href')
                text = tag.get_text(strip=True)
                if href:
                    lab_links.append(href)
                    names.append(text)
                    areas.append(area)  # Optionally store which research area this lab is under
                    html.append(fetch_lab_html(href))

        except Exception as e:
            print(f"Error fetching for area '{area}': {e}")

    html = [sanitize_html(h) for h in html]
    encoded_html = [encode_html(h) for h in html]
    # Convert to DataFrame
    df = pd.DataFrame({
        "Research Area": areas,
        "Lab Name": names,
        "Link": lab_links,
        "HTML": encoded_html
    })

    # Save to CSV
    df.to_csv("../modules/data/labs_with_html.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        lineterminator='\n',
        encoding='utf-8')
    summarize_labs()
    return df

def encode_html(html_string):
    if not isinstance(html_string, str):
        html_string = str(html_string)
    return base64.b64encode(html_string.encode('utf-8')).decode('utf-8')

def fetch_lab_html(lab_link):
    print(f"Getting html for " + lab_link)
    html = ""
    try:
        driver.get(lab_link)
        time.sleep(2)  # let JS load
        html = driver.page_source
        return html
    except Exception as e:
        print(f"Undetected Selenium failed for {lab_link}: {e}")
        return ""

def sanitize_html(raw_html):
    if not isinstance(raw_html, str):
        raw_html = str(raw_html)

    # Replace line breaks, tabs, multiple spaces
    cleaned = raw_html.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    cleaned = re.sub(' +', ' ', cleaned)

    # Replace double quotes with single to avoid CSV quoting issues
    cleaned = cleaned.replace('"', "'")

    return cleaned.strip()

def main():
    try:
        data['research_areas'] = get_research_areas()
        data['institutes_and_centers'] = get_institutes_and_centers()
        data['research_highlights'] = get_current_research_highlights()
        
        data['research_profs'] = {}
        for area_name, area_url in data['research_areas'].items():
            data['research_profs'][area_name] = get_professors_by_area(area_name, area_url)
        save_publications_per_row(data['research_profs'])    
        process_publications()    
        rs_df = get_research_spaces()
        rs_df.to_csv("../modules/data/research_spaces.csv")
        labs_df = get_labs_links()

    except Exception as e:
        print('Error:', e)

    finally:
        # Close and quit the driver at the end
        driver.close()
        driver.quit()

    # Save the organized data to a JSON file
    # with open('../modules/data/data_dump.json', 'w') as outfile:
    #     json.dump(data, outfile, indent=4)
           
if __name__ == "__main__":
    main()
