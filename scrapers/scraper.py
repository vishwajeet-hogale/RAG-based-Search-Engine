import json
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import requests
import pandas as pd
import os

# === Set Up Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "modules", "data")

def data_path(filename):
    return os.path.join(DATA_DIR, filename)

# Load metadata
with open(os.path.join(BASE_DIR, "metadata.json"), 'r') as f:
    metadata = json.load(f)

# Get base URLs from metadata
base_url = metadata["Khoury College of Computer Science"]["Research_landing"]["base_url"]
research_url = metadata["Khoury College of Computer Science"]["Research_areas"]["base_url"]
institutes_and_centers_url = metadata["Khoury College of Computer Science"]["Institutes_and_centers"]["base_url"]
research_spaces_url = metadata["Khoury College of Computer Science"]["Research_spaces"]["base_url"]
labs_url = metadata["Khoury College of Computer Science"]["Labs_groups"]["base_url"]

# Initialize Chrome driver
options = uc.ChromeOptions()
options.headless = True
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = uc.Chrome(options=options)

# Data container
data = {}

def get_research_areas():
    print("Getting research areas")
    driver.get(base_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "main")))
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    research_area_urls = {}
    research_areas = soup.find_all('li', class_='wp-block-khoury-link-list-item')
    
    for research in research_areas:
        area_name = research.text.strip()
        formatted_text = area_name.lower().replace(' ', '-')
        new_url = research_url + formatted_text
        research_area_urls[area_name] = new_url

    df = pd.DataFrame(list(research_area_urls.items()), columns=['Area', 'URL'])
    df.to_csv(data_path("research_areas.csv"), index=False)
    return research_area_urls

def get_institutes_and_centers():
    print("Getting institutes and centers")
    research_data = []

    driver.get(institutes_and_centers_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "main")))
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    institute_names = soup.select("main > div > div")

    for name_tag in institute_names[1:]:
        name = name_tag.find("h3").text.strip()
        description_tag = name_tag.find("p")
        a_element = name_tag.find("a")
        href = a_element['href'] if a_element and a_element.has_attr('href') else ""
        institute_data = {
            "name": name,
            "description": description_tag.text.strip() if description_tag else "",
            "href": href
        }
        research_data.append(institute_data)

    df = pd.DataFrame(research_data)
    df.to_csv(data_path("institutes_and_centers.csv"), index=False)
    return research_data

def get_professor_details(prof_url):
    print(f'Fetching professor details: {prof_url}')
    try:
        driver.get(prof_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "single-people__content")))
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        content_sections = soup.select(".single-people__content div div div div div div")
        prof_data = {"sections": {}}

        for section in content_sections:
            h2_tag = section.find("h2")
            if not h2_tag:
                continue

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

def save_publications_per_row(data, filename):
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

def get_research_spaces():
    print("Getting research spaces")
    research_spaces_df = []
    driver.get(research_spaces_url)
    try:
        research_spaces = WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.wp-block-column.is-layout-flow.wp-block-column-is-layout-flow'))
        )

        for research in research_spaces:
            try:
                title = research.find_element(By.TAG_NAME, 'h3').text
                description = research.find_element(By.TAG_NAME, 'p').text
                link_element = research.find_element(By.TAG_NAME, 'a')
                link = link_element.get_attribute('href')
                research_spaces_df.append([title, description, link])
            except Exception as e:
                print(f"Error extracting a research space: {e}")

    except Exception as e:
        print('Timeout while collecting research area URLs:', e)
    
    return pd.DataFrame(research_spaces_df, columns=["Lab", "Description", "Link"])

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
        profs_list.append({"name": prof_name, "url": prof_url, "info": prof_data})

    return profs_list

def get_current_research_highlights():
    print("Getting research highlights")
    driver.get(base_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "main")))
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    highlight_divs = soup.select("main > div > div:nth-of-type(6) > div:nth-of-type(2) > div > div > div > div")
    data = []

    for div in highlight_divs:
        try:
            h2_text = div.find("h2").text.strip()
        except:
            h2_text = ""
        try:
            p_text = div.find("p").text.strip()
        except:
            p_text = ""
        try:
            href = div.find("a")["href"]
        except:
            href = ""

        data.append({
            "title": h2_text,
            "description": p_text,
            "link": href
        })

    df = pd.DataFrame(data)
    df.to_csv(data_path("current_research_highlights.csv"), index=False)
    return data

def get_labs_links():
    print("Getting labs")
    lab_links, names, areas = [], [], []

    for area, url in data['research_areas'].items():
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
                    areas.append(area)
        except Exception as e:
            print(f"Error fetching for area '{area}': {e}")

    df = pd.DataFrame({
        "Lab Name": names,
        "Link": lab_links,
        "Research Area": areas
    })

    df.to_csv(data_path("labs.csv"), index=False)
    return df

def main():
    try:
        research_areas_file = data_path("research_areas.csv")
        if not os.path.exists(research_areas_file):
            data['research_areas'] = get_research_areas()
        else:
            df = pd.read_csv(research_areas_file)
            data['research_areas'] = dict(zip(df['Area'], df['URL']))

        if not os.path.exists(data_path("institutes_and_centers.csv")):
            data['institutes_and_centers'] = get_institutes_and_centers()

        if not os.path.exists(data_path("current_research_highlights.csv")):
            data['research_highlights'] = get_current_research_highlights()

        professor_info_file = data_path("professor_info.csv")
        if not os.path.exists(professor_info_file):
            data['research_profs'] = {}
            for area_name, area_url in data['research_areas'].items():
                data['research_profs'][area_name] = get_professors_by_area(area_name, area_url)
            save_publications_per_row(data['research_profs'], professor_info_file)

        research_spaces_file = data_path("research_spaces.csv")
        if not os.path.exists(research_spaces_file):
            rs_df = get_research_spaces()
            rs_df.to_csv(research_spaces_file, index=False)

        labs_file = data_path("labs.csv")
        if not os.path.exists(labs_file):
            get_labs_links()

        profs_file = data_path("professors.csv")
        if not os.path.exists(profs_file) and os.path.exists(professor_info_file):
            df = pd.read_csv(professor_info_file)
            df = df[df.columns[1:-4]].drop_duplicates()
            df.to_csv(profs_file, index=False)

    except Exception as e:
        print('Error:', e)

    finally:
        driver.close()
        driver.quit()

    with open(data_path("data_dump.json"), 'w') as outfile:
        json.dump(data, outfile, indent=4)

if __name__ == "__main__":
    main()
