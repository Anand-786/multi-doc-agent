# chunker.py

import json
from bs4 import BeautifulSoup, NavigableString
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "scraped_gem5_docs_with_html.json"
OUTPUT_FILE = "gem5_docs_chunks2.json"

def get_content_until(start_node, stop_tags):
    """
    Collects text from sibling nodes until a stop tag is encountered.
    """
    content = []
    for sibling in start_node.find_next_siblings():
        if sibling.name in stop_tags:
            break
        if isinstance(sibling, NavigableString) and sibling.strip():
            content.append(sibling.strip())
        elif sibling.name:
            content.append(sibling.get_text(separator='\n', strip=True))
    return '\n'.join(content)

def chunk_html_hierarchically(page_data):
    """
    Splits HTML content into semantic chunks based on H2 and H3 tags.
    Each H3 section becomes a chunk, with its parent H2 as context.
    """
    url = page_data['url']
    page_title = page_data.get('title', 'No Title Found')
    html_content = page_data['html_content']

    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = []
    
    # Clean up irrelevant parts of the page first
    for selector in ['div.edit', 'div.navbuttons']:
        for element in soup.select(selector):
            element.decompose()

    # Find all h2 tags to define major sections
    h2_tags = soup.find_all('h2')

    if not h2_tags:
        # If no h2 tags, treat the whole page as a single chunk
        text = soup.get_text(separator='\n', strip=True)
        if text:
            chunk_metadata = {
                "source_url": url,
                "page_title": page_title,
                "parent_section": "Overview",
                "section_heading": Path(url).name,
            }
            chunks.append({"text": text, "metadata": chunk_metadata})
        return chunks

    chunk_counter = 1
    for h2 in h2_tags:
        h2_text = h2.get_text(strip=True)
        
        # Find all h3 tags within the current h2 section
        h3_tags_in_section = []
        for sibling in h2.find_next_siblings():
            if sibling.name == 'h2':
                break
            if sibling.name == 'h3':
                h3_tags_in_section.append(sibling)
        
        # If there are no h3 tags, the entire h2 section is one chunk
        if not h3_tags_in_section:
            section_content = get_content_until(h2, ['h2'])
            full_text = f"{h2_text}\n{section_content}".strip()
            
            chunk_metadata = {
                "source_url": url,
                "page_title": page_title,
                "parent_section": h2_text,
                "section_heading": "Overview",
            }
            chunks.append({"text": full_text, "metadata": chunk_metadata})
            chunk_counter += 1
        else:
            # Create a chunk for each h3 section
            for h3 in h3_tags_in_section:
                h3_text = h3.get_text(strip=True)
                # Content is between this h3 and the next h2 or h3
                h3_content = get_content_until(h3, ['h2', 'h3'])
                
                # Prepend the h2 context for better understanding
                full_text = f"Section: {h2_text}\n\nSub-section: {h3_text}\n\n{h3_content}".strip()

                chunk_metadata = {
                    "source_url": url,
                    "page_title": page_title,
                    "parent_section": h2_text,
                    "section_heading": h3_text,
                }
                chunks.append({"text": full_text, "metadata": chunk_metadata})
                chunk_counter += 1
                
    return chunks

if __name__ == "__main__":
    print(f"Loading raw HTML data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        scraped_data = json.load(f)

    print("Starting hierarchical chunking process (H2 -> H3)...")
    all_chunks = []
    
    for i, page in enumerate(scraped_data):
        print(f"  - Processing page {i+1}/{len(scraped_data)}: {page.get('title', page['url'])}")
        page_chunks = chunk_html_hierarchically(page)
        all_chunks.extend(page_chunks)

    print(f"\n✅ Chunking complete! Created {len(all_chunks)} chunks.")

    print("\n--- Example Chunks ---")
    if all_chunks:
        for chunk in all_chunks[:2]:
            print(f"URL: {chunk['metadata']['source_url']}")
            print(f"PARENT SECTION: {chunk['metadata']['parent_section']}")
            print(f"SECTION HEADING: {chunk['metadata']['section_heading']}")
            print(f"TEXT SNIPPET: {chunk['text'][:200].strip()}...")
            print("--------------------")
    else:
        print("No chunks were generated.")

    print(f"\nSaving chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)
    
    print("✅ Chunks saved successfully.")