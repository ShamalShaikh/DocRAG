"""
Test module for the HTML to Markdown converter.
"""

import pytest
from bs4 import BeautifulSoup
from html_to_markdown_converter import HTMLToMarkdownConverter, create_converter

# Sample HTML content for testing
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="author" content="John Doe">
    <meta name="publication_date" content="2024-03-15">
    <title>Test Article</title>
</head>
<body>
    <article>
        <h1>Sample Article</h1>
        <p>This is a test article with some <strong>bold text</strong> and a <a href="https://example.com">link</a>.</p>
        <h2>Section 1</h2>
        <p>Here's a list of items:</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
        </ul>
        <h2>References</h2>
        <p>Some references here.</p>
        <h2>External Links</h2>
        <p>Some external links here.</p>
    </article>
    <script>console.log("This should be removed");</script>
</body>
</html>
"""

@pytest.fixture
def converter():
    """Create a converter instance for testing."""
    return create_converter()

def test_html_cleaning(converter):
    """Test HTML cleaning functionality."""
    cleaned = converter._clean_html(SAMPLE_HTML)
    assert 'script' not in cleaned
    assert 'console.log' not in cleaned
    assert 'Sample Article' in cleaned

def test_metadata_extraction(converter):
    """Test metadata extraction."""
    soup = BeautifulSoup(SAMPLE_HTML, 'html.parser')
    metadata = converter._extract_metadata(soup)
    
    assert metadata['author'] == 'John Doe'
    assert metadata['publication_date'] == '2024-03-15'

def test_full_conversion(converter):
    """Test the complete conversion process."""
    result = converter.convert(SAMPLE_HTML)
    
    assert isinstance(result, dict)
    assert 'markdown' in result
    assert 'metadata' in result
    assert 'tags' in result
    
    # Check metadata
    assert result['metadata']['author'] == 'John Doe'
    assert result['metadata']['publication_date'] == '2024-03-15'
    
    # Check markdown content
    markdown = result['markdown']
    assert '# Sample Article' in markdown
    assert '[link](https://example.com)' in markdown
    assert '## Section 1' in markdown
    
    # Check word count and reading time
    assert result['metadata']['word_count'] > 0
    assert result['metadata']['reading_time'] > 0
    
    # Check tags
    assert isinstance(result['tags'], list)
    assert len(result['tags']) > 0

def test_error_handling(converter):
    """Test error handling for invalid input."""
    with pytest.raises(ValueError):
        converter.convert("")
    
    with pytest.raises(Exception):
        converter.convert("Invalid HTML<") 