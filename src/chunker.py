import json
from bs4 import BeautifulSoup
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "scraped_gem5_docs_with_html.json"
OUTPUT_FILE = "gem5_docs_chunks.json"

def chunk_html_by_h2(page_data):
    """
    Splits the HTML content of a page into semantic chunks based on H2 tags.
    """
    url = page_data['url']
    page_title = page_data['title']
    html_content = page_data['html_content']

    soup = BeautifulSoup(html_content, 'html.parser')
    
    chunks = []
    
    # Find all h2 tags, which we'll use as the main section dividers
    h2_tags = soup.find_all('h2')

    if not h2_tags:
        # If a page has no h2 tags, treat the whole content as a single chunk
        # after cleaning out unwanted divs like the 'edit' button.
        edit_div = soup.find('div', class_='edit')
        if edit_div:
            edit_div.decompose() # Removes the "Edit this page" div
            
        nav_buttons = soup.find('div', class_='navbuttons')
        if nav_buttons:
            nav_buttons.decompose() # Removes the PREVIOUS/NEXT buttons

        text = soup.get_text(separator='\n', strip=True)
        if text: # Make sure there's actual text to add
            chunk_metadata = {
                "source_url": url,
                "page_title": page_title if page_title != "No Title Found" else Path(url).name,
                "section_heading": "Overview",
                "chunk_num": 1
            }
            chunks.append({"text": text, "metadata": chunk_metadata})
        return chunks

    # Process each section defined by an h2 tag
    for i, h2 in enumerate(h2_tags):
        heading_text = h2.get_text(strip=True)
        
        # Collect all content between this h2 and the next h2
        content_tags = []
        for sibling in h2.find_next_siblings():
            if sibling.name == 'h2':
                break  # Stop when we hit the next section
            content_tags.append(sibling)
        
        # Combine the text from the heading and the collected tags
        content_text = '\n'.join(tag.get_text(separator='\n', strip=True) for tag in content_tags)
        full_chunk_text = f"{heading_text}\n{content_text}".strip()

        chunk_metadata = {
            "source_url": url,
            "page_title": page_title if page_title != "No Title Found" else Path(url).name,
            "section_heading": heading_text,
            "chunk_num": i + 1
        }
        
        chunks.append({"text": full_chunk_text, "metadata": chunk_metadata})
        
    return chunks

if __name__ == "__main__":
    print(f"Loading raw HTML data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        scraped_data = json.load(f)

    print("Starting chunking process using HTML tags...")
    all_chunks = []
    
    for page in scraped_data:
        page_chunks = chunk_html_by_h2(page)
        all_chunks.extend(page_chunks)

    print(f"✅ Chunking complete! Created {len(all_chunks)} chunks.")

    print("\n--- Example Chunks ---")
    for chunk in all_chunks[:2]:
        print(f"URL: {chunk['metadata']['source_url']}")
        print(f"SECTION: {chunk['metadata']['section_heading']}")
        print(f"TEXT SNIPPET: {chunk['text'][:150].strip()}...")
        print("--------------------")

    print(f"\nSaving chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)
    
    print("✅ Chunks saved successfully.")