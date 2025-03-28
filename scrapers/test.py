import json
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
research_spaces_url = metadata["Khoury College of Computer Science"]["Research_spaces"]["base_url"]
labs_url = metadata["Khoury College of Computer Science"]["Labs_groups"]["base_url"]

# Store the results
data = {}

# Function to get all research area URLs
def get_research_areas():
    research_area_urls = {}
    driver.get(base_url)
    
    try:
        research_areas = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'wp-block-khoury-link-list-item'))
        )
        
        for research in research_areas:
            formatted_text = research.text.lower().replace(' ', '-')
            area_name = research.text  # Keep the original name for JSON key
            new_url = research_url + formatted_text
            research_area_urls[area_name] = new_url
        
    except Exception as e:
        print('Error while collecting research area URLs:', e)
    
    return research_area_urls

# Function to get professors by research area
def get_professors_by_area(area_name, area_url):
    print(f'Navigating to: {area_url}')
    driver.get(area_url)
    profs_list = []

    try:
        profs = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, "//section/ul/li/article/div/h3/a"))
        )
        
        for prof in profs:
            prof_name = prof.text
            prof_url = prof.get_attribute('href')
            # print('Prof:', prof_name, '| URL:', prof_url)
            
            # Store in a dictionary
            profs_list.append({
                "name": prof_name,
                "url": prof_url
            })
    
    except Exception as inner_e:
        print('Error accessing new page:', inner_e)
    
    return profs_list

# Function to get all research spaces URLs
def get_research_spaces():
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

def main():
    try:
        # Step 1: Get all research area URLs
        research_area_urls = get_research_areas()

        # Step 2: Visit each research area and get professor details
        for area_name, area_url in research_area_urls.items():
            data[area_name] = get_professors_by_area(area_name, area_url)
            
        
        rs_df = get_research_spaces()
        rs_df.to_csv("Research_spaces.csv")

    except Exception as e:
        print('Error:', e)

    finally:
        # Close and quit the driver at the end
        driver.close()
        driver.quit()

    # Save the organized data to a JSON file
    with open('research_area_professors.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)
        
        
        
if __name__ == "__main__":
    main()

