import json
import requests
from bs4 import BeautifulSoup

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
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    research_area_urls = {}
    research_areas = soup.find_all('li', class_='wp-block-khoury-link-list-item')
    
    for research in research_areas:
        area_name = research.text.strip()
        formatted_text = area_name.lower().replace(' ', '-')
        new_url = research_url + formatted_text
        research_area_urls[area_name] = new_url

    return research_area_urls

# Function to get institutes and centers
def get_institutes_and_centers():
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    res = []
    
    # Select all <li> elements inside <ul> within <main> > <div>
    list_items = soup.select("main > div > ul > li")

    for li in list_items:
        try:
            # Extract <span> text
            span_element = li.find("span")
            span_text = span_element.text.strip() if span_element else "No span text"

            # Extract <a> tag and its href
            a_element = li.find("a")
            href = a_element['href'] if a_element and a_element.has_attr('href') else "No href"

            res.append({"title": span_text, "url": href})
        
        except Exception as e:
            print(f"Error processing list item: {e}")

    return res


def get_professor_details(prof_url):
    """Fetch and parse a professor's page to extract details from collapsible sections."""
    # print(f'Fetching professor details: {prof_url}')
    
    try:
        response = requests.get(prof_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate the collapsible sections inside "single-people__content"
        content_sections = soup.select(".single-people__content div div div div div div")

        prof_data = {
            "sections": {}
        }

        for section in content_sections:
            h2_tag = section.find("h2")
            if not h2_tag:
                continue  # Skip if no header found

            section_name = h2_tag.text.strip()

            if section_name == "Research interests" or section_name == "Education":
                # Extract bullet points
                items = [li.text.strip() for li in section.select("ul li")]
                prof_data["sections"][section_name] = items

            elif section_name == "Biography":
                # Extract biography text
                bio_text = section.find("p").text.strip() if section.find("p") else "No biography available"
                prof_data["sections"][section_name] = bio_text

            elif section_name == "Recent publications":
                # Extract publication details
                publications = []
                for pub in section.select("ul li"):
                    time_tag = pub.find("time")
                    date = time_tag.text.strip() if time_tag else "Unknown Date"

                    a_tag = pub.find("a")
                    pub_name = a_tag.text.strip() if a_tag else "Unknown Publication"
                    pub_link = a_tag["href"] if a_tag and a_tag.has_attr("href") else ""

                    publications.append({"date": date, "publication": pub_name, "link": pub_link})

                prof_data["sections"][section_name] = publications

        return prof_data

    except Exception as e:
        print(f"Error fetching professor details: {e}")
        return {}


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

# Function to get current research highlights
import requests
from bs4 import BeautifulSoup

def get_current_research_highlights():
    response = requests.get(base_url)  # Fetch the webpage
    soup = BeautifulSoup(response.text, "html.parser")  # Parse HTML

    # Locate all target divs under /html/body/main/div/div[6]/div[2]/div/div/div/div
    highlight_divs = soup.select("main > div > div:nth-of-type(5) > div:nth-of-type(2) > div > div > div > div")
    # print(highlight_divs)
    data = []

    for div in highlight_divs:
        try:
            # Extract H2 text (Research highlight title)
            h2_element = div.find("h2")
            h2_text = h2_element.text.strip() if h2_element else ""
        except:
            h2_text = ""

        try:
            # Extract paragraph text (Research description)
            p_element = div.find("p")
            p_text = p_element.text.strip() if p_element else ""
        except:
            p_text = ""

        try:
            # Extract the first hyperlink
            link_element = div.find("a")
            href = link_element["href"] if link_element and link_element.has_attr("href") else ""
        except:
            href = ""

        # Append data in dictionary format
        data.append({
            "title": h2_text,
            "description": p_text,
            "link": href
        })

    return data

# Main execution
try:
    data['research_areas'] = get_research_areas()
    data['institutes_and_centers'] = get_institutes_and_centers()
    data['research_highlights'] = get_current_research_highlights()
    
    data['research_profs'] = {}
    for area_name, area_url in data['research_areas'].items():
        data['research_profs'][area_name] = get_professors_by_area(area_name, area_url)

except Exception as e:
    print('Error:', e)

# Save the organized data to a JSON file
with open('data_dump.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)
    