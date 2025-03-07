import json
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

def get_institutes_and_centers():
    driver.get(base_url)
    res = []
    try:
        # Wait for the <ul> element to be present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "/html/body/main/div/ul"))
        )

        # Find all <li> elements that contain an <a> tag at any level
        list_items = driver.find_elements(By.XPATH, "/html/body/main/div/ul/li[.//a]")

        # Extract and print text from <a> and <span> inside each <li>
        for li in list_items:
            try:
                span_text = li.find_element(By.TAG_NAME, "span").text  # Extract span text
            except:
                span_text = "No span text"  # Handle cases where <span> is missing
            
            try:
                link_element = li.find_element(By.TAG_NAME, "a")  # Find <a> tag
                href = link_element.get_attribute("href")  # Extract href attribute
            except:
                href = "No href"  # Handle cases where <a> is missing

            print(f"Span Text: {span_text}, Href: {href}")
            res.append({
                "title": span_text,
                "url": href
            })

    except Exception as e:
        print("Error:", e)
    
    return res

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

def get_current_research_hightlights():
    # Locate all target divs under the given XPath
    driver.get(base_url)
    div_elements = driver.find_elements(By.XPATH, "/html/body/main/div/div[6]/div[2]/div/div/div/div")
    # Dictionary to store extracted data
    data = []

    for _, div in enumerate(div_elements):
        try:
            # Extract h2 text
            h2_text = div.find_element(By.TAG_NAME, "h2").text.strip()
        except:
            h2_text = ""

        try:
            # Click on the card header or toggle button to expand the card
            toggle_button = div.find_element(By.XPATH, ".//button")  # Adjust XPath if needed
            toggle_button.click()

            # Wait for the content to become visible
            WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.XPATH, ".//p"))
            )

            # Extract p text
            p_text = div.find_element(By.XPATH, ".//p").text.strip()
        except:
            p_text = ""

        try:
            # Extract href (assuming it's inside an <a> tag somewhere in the div)
            link_element = div.find_element(By.XPATH, ".//a")  # Finds first <a> inside div
            href = link_element.get_attribute("href")
        except:
            href = ""

        # Store in dictionary using h2 as the key
        data.append({"title":h2_text,"description": p_text, "link": href})
  
    return data

try:
    # Step 1: Get all research area URLs
    research_area_urls = get_research_areas()
    data['research_areas'] = research_area_urls
    data['institutes_and_centers'] = get_institutes_and_centers()
    data['research_hightlights'] = get_current_research_hightlights()
    # Step 2: Visit each research area and get professor details
    data['research_profs'] = {}
    for area_name, area_url in research_area_urls.items():
        data['research_profs'][area_name] = get_professors_by_area(area_name, area_url)

except Exception as e:
    print('Error:', e)

finally:
    # Close and quit the driver at the end
    driver.close()
    driver.quit()

# Save the organized data to a JSON file
with open('data_dump.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)