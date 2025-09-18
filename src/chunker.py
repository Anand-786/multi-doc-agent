import json
from bs4 import BeautifulSoup, NavigableString
from pathlib import Path

INPUT_FILE = "scraped_gem5_docs_with_html.json"
OUTPUT_FILE = "gem5_docs_chunks2.json"

def get_content_until(start_node, stop_tags):
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
    url = page_data['url']
    page_title = page_data.get('title', 'No Title Found')
    html_content = page_data['html_content']

    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = []
    
    for selector in ['div.edit', 'div.navbuttons']:
        for element in soup.select(selector):
            element.decompose()

    h2_tags = soup.find_all('h2')

    if not h2_tags:
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
        
        h3_tags_in_section = []
        for sibling in h2.find_next_siblings():
            if sibling.name == 'h2':
                break
            if sibling.name == 'h3':
                h3_tags_in_section.append(sibling)
        
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
            for h3 in h3_tags_in_section:
                h3_text = h3.get_text(strip=True)
                h3_content = get_content_until(h3, ['h2', 'h3'])
                
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

    print("Starting hierarchical chunking process...")
    all_chunks = []
    
    for i, page in enumerate(scraped_data):
        print(f"- Processing page {i+1}/{len(scraped_data)}: {page.get('title', page['url'])}")
        page_chunks = chunk_html_hierarchically(page)
        all_chunks.extend(page_chunks)

    print(f"\nChunking complete! Created {len(all_chunks)} chunks.")

    print("\n-Example Chunks-")
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
    
    print("Chunks saved successfully.")