"""
Test module for the WebScraper class, focusing on Wikipedia page scraping functionality.

This module contains test cases that verify the WebScraper's ability to:
1. Extract data from Wikipedia pages using HTML scraping
2. Handle various edge cases and errors
3. Parse specific Wikipedia page elements correctly
"""

import pytest
from bs4 import BeautifulSoup
from typing import Dict, Any
import re
import logging

from web_scraper import WebScraper, ScrapingConfig, create_default_scraper

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test URLs and expected data
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Web_scraping"
WIKIPEDIA_API_BASE = "https://en.wikipedia.org/w/api.php"

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace, citations, and other noise.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove citation numbers [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def debug_print_element(element, prefix=""):
    """Helper function to print element details for debugging."""
    if element:
        logger.debug(f"{prefix} Tag: {element.name}")
        logger.debug(f"{prefix} Classes: {element.get('class', [])}")
        logger.debug(f"{prefix} ID: {element.get('id', '')}")
        logger.debug(f"{prefix} Text: {element.get_text().strip()[:100]}")  # First 100 chars

def normalize_section_name(text: str) -> str:
    """
    Normalize a section name for consistent comparison.
    
    Args:
        text (str): Raw section name text
        
    Returns:
        str: Normalized section name
    """
    # Remove any citation numbers and extra whitespace
    text = re.sub(r'\[\d+\]', '', text)
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    # Remove any special characters
    text = re.sub(r'[^\w\s-]', '', text)
    return text

def extract_sections(soup: BeautifulSoup) -> list:
    """
    Extract section titles from a Wikipedia page using multiple methods.
    
    This function tries multiple approaches to find section titles:
    1. Direct search for all section headings
    2. Fallback to table of contents
    3. Final fallback to standard section names
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        list: List of section titles
    """
    sections = {}  # Using dict to store both original and normalized names
    
    # Define standard sections with their normalized forms
    standard_sections = {
        'See also': 'see also',
        'References': 'references',
        'External links': 'external links',
        'Notes': 'notes',
        'Bibliography': 'bibliography'
    }
    
    logger.debug("Starting section extraction...")
    
    # Method 1: Direct search for all section headings
    headlines = soup.find_all(['h2', 'h3'])  # Include both h2 and h3 headings
    logger.debug(f"Found {len(headlines)} headline elements")
    
    for heading in headlines:
        section_text = None
        
        # Try mw-headline span
        headline_span = heading.find('span', {'class': 'mw-headline'})
        if headline_span:
            section_text = headline_span.get_text().strip()
        else:
            # Otherwise, get the text from the heading itself
            section_text = heading.get_text().strip()
        
        if section_text:
            normalized_text = normalize_section_name(section_text)
            logger.debug(f"Candidate section - Original: '{section_text}', Normalized: '{normalized_text}'")
            sections[normalized_text] = section_text
    
    # Method 2: Try extracting from the table of contents if no headings found
    if not sections:
        toc = soup.find('div', {'id': 'toc'})
        if toc:
            logger.debug("Found table of contents")
            for link in toc.find_all(['span', 'a'], {'class': ['toctext', 'tocnumber']}):
                section_text = link.get_text().strip()
                if section_text:
                    normalized_text = normalize_section_name(section_text)
                    logger.debug(f"TOC candidate - Original: '{section_text}', Normalized: '{normalized_text}'")
                    sections[normalized_text] = section_text
    
    # Method 3: Look for specific section markers via IDs or direct text search
    for marker in ['External_links', 'References', 'See_also']:
        section = soup.find(id=marker)
        if section:
            heading = section.find_parent(['h2', 'h3'])
            if heading:
                section_text = heading.get_text().strip()
                normalized_text = normalize_section_name(section_text)
                logger.debug(f"Found section by ID '{marker}' - Original: '{section_text}', Normalized: '{normalized_text}'")
                sections[normalized_text] = section_text
        nav_template = soup.find('span', {'class': 'mw-headline', 'id': marker})
        if nav_template:
            section_text = nav_template.get_text().strip()
            normalized_text = normalize_section_name(section_text)
            logger.debug(f"Found section by nav template for '{marker}' - Original: '{section_text}', Normalized: '{normalized_text}'")
            sections[normalized_text] = section_text
    
    # Method 4: Final fallback to explicitly check for standard sections
    for std_section, std_normalized in standard_sections.items():
        if std_normalized not in sections:
            # Search for any element whose normalized text matches the standard section
            potential_elements = soup.find_all(['span', 'a', 'h2', 'h3'])
            for element in potential_elements:
                text = element.get_text().strip()
                if text and normalize_section_name(text) == std_normalized:
                    logger.debug(f"Explicitly found standard section: '{std_section}' via element text match")
                    sections[std_normalized] = std_section
                    break
    
    # Final explicit check: if "external links" is still missing, search for it directly
    if 'external links' not in sections:
        # Look for any element that contains "External links" exactly (case-insensitive)
        candidate = soup.find(string=lambda t: t and "external links" in t.lower())
        if candidate:
            parent_heading = candidate.find_parent(['h2', 'h3'])
            if parent_heading:
                section_text = parent_heading.get_text().strip()
                normalized_text = normalize_section_name(section_text)
                if normalized_text == 'external links':
                    logger.debug("Explicitly added 'External links' from direct search")
                    sections[normalized_text] = section_text
    
    final_sections = list(sections.values())
    logger.debug(f"Final sections found: {final_sections}")
    return final_sections

def wikipedia_parser(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Custom parser function for Wikipedia pages.
    
    This function extracts key information from Wikipedia pages using a more robust
    approach that handles various page structures and nested elements.
    
    Strategy:
    1. Title: Extract from title tag and remove " - Wikipedia" suffix
    2. First Paragraph: Look for the first substantial paragraph in the main content
       area, skipping disambiguation notices and other non-content elements
    3. Sections: Extract all section headings using multiple methods
    4. References: Count all citation elements
    
    Args:
        soup (BeautifulSoup): Parsed HTML content of the Wikipedia page
        
    Returns:
        Dict[str, Any]: Extracted data including title, first paragraph, and sections
    """
    data = {
        'title': None,
        'first_paragraph': None,
        'sections': [],
        'references_count': 0
    }
    
    # Debug: Print the overall structure
    logger.debug("Starting Wikipedia page parsing")
    logger.debug(f"Title tag found: {bool(soup.title)}")
    
    # Extract title (without " - Wikipedia" suffix)
    if soup.title:
        title = soup.title.string
        if title and " - Wikipedia" in title:
            data['title'] = title.replace(" - Wikipedia", "").strip()
            logger.debug(f"Extracted title: {data['title']}")
    
    # Find the main content area
    content_div = soup.find('div', {'class': 'mw-parser-output'})
    if content_div:
        logger.debug("Found main content area")
        # Find all paragraphs in the content area
        paragraphs = content_div.find_all('p', recursive=False)
        logger.debug(f"Found {len(paragraphs)} top-level paragraphs")
        
        # Look for the first substantial paragraph
        for p in paragraphs:
            # Skip empty paragraphs or those with specific classes
            if not p.get('class') or not any(c in ['mw-empty-elt'] for c in p.get('class', [])):
                text = clean_text(p.get_text())
                # Check if paragraph has meaningful content
                if text and len(text) > 50:
                    data['first_paragraph'] = text
                    logger.debug(f"Found first paragraph: {text[:100]}...")  # First 100 chars
                    break
    
    # Extract sections
    data['sections'] = extract_sections(soup)
    
    # Count references
    references = soup.find_all(['cite', 'span'], class_=['citation', 'reference-text'])
    if not references:  # Fallback to reference list items
        references = soup.find_all('li', class_='reference')
    data['references_count'] = len(references)
    logger.debug(f"Found {data['references_count']} references")
    
    return data

@pytest.fixture
def scraper():
    """Fixture to create a WebScraper instance for testing."""
    config = ScrapingConfig(
        use_api=False,  # Using HTML scraping for Wikipedia
        timeout=30,
        max_retries=2
    )
    return WebScraper(config)

def test_wikipedia_scraping_basic(scraper):
    """
    Test basic Wikipedia page scraping functionality.
    
    Verifies that the scraper can:
    1. Successfully connect to Wikipedia
    2. Extract the correct page title
    3. Extract non-empty content
    """
    result = scraper.scrape_html(WIKIPEDIA_URL, wikipedia_parser)
    
    assert result is not None, "Scraping result should not be None"
    assert result['title'] == "Web scraping", "Page title should be 'Web scraping'"
    assert result['first_paragraph'], "First paragraph should not be empty"
    assert len(result['sections']) > 0, "Page should have at least one section"
    assert result['references_count'] > 0, "Page should have at least one reference"

def test_wikipedia_content_validation(scraper):
    """
    Test that scraped Wikipedia content contains expected information.
    
    Verifies specific content elements that should be present in the
    Web scraping Wikipedia page.
    """
    result = scraper.scrape_html(WIKIPEDIA_URL, wikipedia_parser)
    
    # Check for expected sections
    expected_sections = {"See also", "References", "External links"}
    found_sections = set(result['sections'])
    
    # Debug output for section comparison
    logger.debug("Section comparison:")
    logger.debug(f"Expected sections: {expected_sections}")
    logger.debug(f"Found sections: {found_sections}")
    logger.debug(f"Missing sections: {expected_sections - found_sections}")
    
    assert expected_sections.issubset(found_sections), \
        f"Not all expected sections were found. Missing: {expected_sections - found_sections}"
    
    # Verify first paragraph contains key terms
    first_para = result['first_paragraph'].lower()
    assert "web scraping" in first_para, \
        "First paragraph should contain 'web scraping'"
    assert "data" in first_para, \
        "First paragraph should mention 'data'"

def test_parallel_wikipedia_scraping(scraper):
    """
    Test parallel scraping of multiple Wikipedia pages.
    
    Verifies that the scraper can handle multiple requests simultaneously
    and correctly aggregate the results.
    """
    urls = [
        "https://en.wikipedia.org/wiki/Web_scraping",
        "https://en.wikipedia.org/wiki/Data_extraction",
        "https://en.wikipedia.org/wiki/Web_crawler"
    ]
    
    results = scraper.scrape_urls_parallel(urls, wikipedia_parser)
    
    assert len(results) == len(urls), \
        "Should have results for all URLs"
    
    for url, result in results.items():
        assert result is not None, f"Result for {url} should not be None"
        assert result['title'], f"Title should be present for {url}"
        assert result['first_paragraph'], \
            f"First paragraph should be present for {url}"

def test_error_handling(scraper):
    """
    Test error handling for various edge cases.
    
    Verifies that the scraper properly handles:
    1. Invalid URLs
    2. Network timeouts
    3. Malformed HTML responses
    """
    # Test invalid URL
    with pytest.raises(Exception):
        scraper.scrape_html(
            "https://en.wikipedia.org/wiki/NonexistentPage123456789",
            wikipedia_parser
        )
    
    # Test parallel scraping with mix of valid and invalid URLs
    urls = [
        WIKIPEDIA_URL,
        "https://en.wikipedia.org/wiki/NonexistentPage123456789",
        "https://en.wikipedia.org/wiki/Data_extraction"
    ]
    
    results = scraper.scrape_urls_parallel(urls, wikipedia_parser)
    
    assert results[WIKIPEDIA_URL] is not None, \
        "Valid URL should return results"
    assert any(result is None for result in results.values()), \
        "Invalid URL should return None"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 