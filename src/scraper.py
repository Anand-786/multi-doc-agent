import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

BASE_URL = "https://www.gem5.org/"
START_URL = "https://www.gem5.org/documentation/"
EXCLUSION_KEYWORDS = ["kvm", "sphinx", "doxygen", "debugging"]
OUTPUT_FILE = "scraped_gem5_docs_with_html.json"

def get_all_doc_links(start_url):
    print("Phase 1: Collecting all documentation links...")
    try:
        response = requests.get(start_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the start URL: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    sidenav = soup.find('div', class_='sidenav')
    if not sidenav:
        print("Error: Could not find the sidebar navigation.")
        return []

    doc_links = set()
    for a_tag in sidenav.find_all('a', href=True):
        link_text = a_tag.get_text(strip=True)
        href = a_tag['href']
        
        if '#' not in href and not any(keyword in link_text.lower() or keyword in href.lower() for keyword in EXCLUSION_KEYWORDS):
            full_url = urljoin(BASE_URL, href)
            doc_links.add(full_url)
    
    print(f"Found {len(doc_links)} valid links to scrape.")
    return sorted(list(doc_links))

def scrape_page_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        time.sleep(0.5)
    except requests.exceptions.RequestException as e:
        print(f"\nWarning: Failed to scrape {url}. Reason: {str(e)[:100]}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    content_div = soup.find('div', id='doc-container')
    if not content_div:
        return None

    title_tag = content_div.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "No Title Found"
    
    html_content = str(content_div)
    
    return {
        "url": url,
        "title": title,
        "html_content": html_content
    }

if __name__ == "__main__":
    all_links = get_all_doc_links(START_URL)
    
    if all_links:
        print("\n- Links to be Scraped -")
        for link in all_links:
            print(link)
        print("---------------------------\n")

        print(f"Phase 2: Scraping content from {len(all_links)} pages...")
        scraped_data = []
        
        total_pages = len(all_links)
        for i, link in enumerate(all_links):
            print(f"Scraping [{i+1}/{total_pages}] {link} ... ", end="", flush=True)
            
            data = scrape_page_content(link)
            
            if data:
                scraped_data.append(data)
                print("Done.")
            else:
                print("Failed.")

        print(f"\nScraping complete! Successfully scraped {len(scraped_data)} pages.")
        
        if scraped_data:
            print(f"Saving scraped data to {OUTPUT_FILE}...")
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, indent=4)
            print("Data saved successfully.")
        else:
            print("No data was scraped. The output file will not be created.")