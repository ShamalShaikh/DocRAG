"""
HTML to Markdown Converter Module

This module provides functionality to convert HTML content to Markdown format,
extract metadata, generate summaries, and apply tagging using LLM capabilities.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import json
from dataclasses import dataclass
from groq_client import create_groq_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConversionResult:
    """Data class to store conversion results."""
    markdown: str
    author: Optional[str]
    publication_date: Optional[str]
    summary: Optional[str]
    tags: List[str]
    word_count: int
    reading_time: int  # in minutes

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            'markdown': self.markdown,
            'metadata': {
                'author': self.author,
                'publication_date': self.publication_date,
                'word_count': self.word_count,
                'reading_time': self.reading_time
            },
            'summary': self.summary,
            'tags': self.tags
        }

class HTMLToMarkdownConverter:
    """
    A class to convert HTML content to Markdown with enhanced metadata extraction
    and content analysis using LLM capabilities.
    """

    def __init__(self, 
                 model_name: str = "gemma2-9b-it",
                 summary_threshold: int = 500,  # words
                 max_summary_length: int = 150):  # words
        """
        Initialize the converter with specified parameters.

        Args:
            model_name: Name of the Groq model to use
            summary_threshold: Word count threshold for generating summaries
            max_summary_length: Maximum length of generated summaries in words
        """
        self.model_name = model_name
        self.summary_threshold = summary_threshold
        self.max_summary_length = max_summary_length
        self.llm = create_groq_client(model_name=model_name)
        logger.info(f"Initialized converter with model: {model_name}")

    def _clean_html(self, html: str) -> str:
        """
        Clean HTML content by removing scripts, styles, and unnecessary elements.

        Args:
            html: Raw HTML string

        Returns:
            Cleaned HTML string
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'iframe', 'nav', 'footer']):
                element.decompose()
            
            return str(soup)
        except Exception as e:
            logger.error(f"Error cleaning HTML: {str(e)}")
            raise

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract metadata from HTML content.

        Args:
            soup: BeautifulSoup object of the HTML content

        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            'author': None,
            'publication_date': None
        }

        try:
            # Try meta tags first
            meta_author = soup.find('meta', {'name': ['author', 'article:author']})
            if meta_author:
                metadata['author'] = meta_author.get('content')

            meta_date = soup.find('meta', {
                'name': ['date', 'article:published_time', 'publication_date']
            })
            if meta_date:
                metadata['publication_date'] = meta_date.get('content')

            # Try schema.org markup
            article = soup.find(['article', 'div'], {'itemtype': 'http://schema.org/Article'})
            if article:
                author = article.find(['span', 'div'], {'itemprop': 'author'})
                if author:
                    metadata['author'] = author.get_text().strip()

                date = article.find(['time', 'span'], {'itemprop': 'datePublished'})
                if date:
                    metadata['publication_date'] = date.get('datetime', date.get_text().strip())

            # Try common HTML patterns if metadata is still missing
            if not metadata['author']:
                author_candidates = soup.find_all(['a', 'span', 'div'], 
                    {'class': re.compile(r'author|byline', re.I)})
                if author_candidates:
                    metadata['author'] = author_candidates[0].get_text().strip()

            if not metadata['publication_date']:
                date_candidates = soup.find_all(['time', 'span', 'div'],
                    {'class': re.compile(r'date|published|time', re.I)})
                if date_candidates:
                    metadata['publication_date'] = date_candidates[0].get_text().strip()

        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")

        return metadata

    def _convert_to_markdown(self, html: str) -> str:
        """
        Convert HTML to Markdown using LLM.

        Args:
            html: Cleaned HTML content

        Returns:
            Markdown formatted string
        """
        try:
            #TODO: Make this prompt stick to the topic, and be more robust and handle more edge cases. 
            prompt = f"""
            1. Convert <h1> to # Heading, <h2> to ## Heading, <h3> to ### Heading, etc. Convert <p> to paragraphs, <ul> / <li> to bullet lists, <strong> to **bold**, and other HTML elements to their corresponding Markdown equivalents.
            2. Important: Keep atleast one '#' for main heading, and atleast one '##' for subheadings in the result.
            3. Note: Keep in mind that some HTML elements (like CSS classes or IDs) may not have direct Markdown equivalents. Also, do not wrap your result in triple backticks or code fences.
            4. Don't leave any HTML tags in the result.
            
            HTML:
            {html}
            """

            response = self.llm.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            
            markdown = response.response.strip()
            logger.info(f"Successfully converted HTML to Markdown: {markdown}")
            return markdown

        except Exception as e:
            logger.error(f"Error converting to Markdown: {str(e)}")
            raise

    def _generate_summary(self, text: str) -> Optional[str]:
        """
        Generate a summary of the content using LLM if it exceeds the threshold.

        Args:
            text: Text content to summarize

        Returns:
            Generated summary or None if text is below threshold
        """
        word_count = len(text.split())
        if word_count < self.summary_threshold:
            return None

        try:
            prompt = f"""
            Summarize the following text in a concise manner, capturing the main points
            and key information. Keep the summary under {self.max_summary_length} words.

            Text:
            {text}
            """

            response = self.llm.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            
            summary = response.response.strip()
            logger.info(f"Generated summary of length: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.warning(f"Error generating summary: {str(e)}")
            return None

    def _generate_tags(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Generate relevant tags for the content using LLM.

        Args:
            text: Content text
            metadata: Extracted metadata

        Returns:
            List of generated tags
        """
        try:
            prompt = f"""
            Generate 5-8 relevant tags for the following content. 
            Consider the topic, domain, and key concepts discussed.
            Return the tags as a comma-separated list.

            Content:
            {text[:1000]}...  # Using first 1000 chars for tag generation

            Metadata:
            {json.dumps(metadata, indent=2)}
            """

            response = self.llm.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            
            tags = [tag.strip() for tag in response.response.split(',')]
            logger.info(f"Generated {len(tags)} tags")
            return tags

        except Exception as e:
            logger.warning(f"Error generating tags: {str(e)}")
            return []

    def convert(self, html: str) -> Dict[str, Any]:
        """
        Convert HTML to Markdown with metadata extraction.
        
        Args:
            html: HTML content to convert
            
        Returns:
            Dictionary containing:
            - markdown: Converted Markdown text
            - metadata: Extracted metadata
            - summary: Optional summary if content is long enough
            - tags: Extracted tags/keywords
            
        Raises:
            ValueError: If HTML content is empty or invalid
            RuntimeError: If conversion fails
        """
        if not html or not html.strip():
            raise ValueError("HTML content cannot be empty")
            
        try:
            # Clean HTML
            cleaned_html = self._clean_html(html)
            
            # Split content into smaller chunks if needed
            soup = BeautifulSoup(cleaned_html, 'html.parser')
            
            # Extract metadata first
            metadata = self._extract_metadata(soup)
            
            # Process content in chunks if it's too large
            chunks = []
            current_chunk = []
            current_length = 0
            max_chunk_length = 4000  # Keep chunks under 4000 tokens to stay within limits
            
            # Process each top-level element
            for element in soup.find_all(recursive=False):
                element_text = str(element)
                element_length = len(element_text.split())
                
                if current_length + element_length > max_chunk_length:
                    # Current chunk is full, process it
                    if current_chunk:
                        chunk_html = "".join(current_chunk)
                        try:
                            # Create a clear prompt for HTML to Markdown conversion
                            prompt = f"""Convert the following HTML to clean, well-formatted Markdown.
Follow these rules:
1. Preserve headings, lists, links, and basic formatting
2. Remove any unnecessary HTML attributes or styling
3. Keep the document structure intact
4. Use proper Markdown syntax for links, images, and formatting

HTML Content:
{chunk_html}

Convert the above HTML to Markdown format."""

                            # Call Groq API with proper message formatting
                            response = self.llm.generate_text(prompt)
                            chunks.append(response["text"].strip())
                        except Exception as e:
                            logger.warning(f"Error converting chunk: {str(e)}")
                            # Try to process the chunk with simpler conversion
                            from html2text import HTML2Text
                            h = HTML2Text()
                            h.ignore_links = False
                            chunks.append(h.handle(chunk_html))
                    
                    # Start new chunk
                    current_chunk = [element_text]
                    current_length = element_length
                else:
                    current_chunk.append(element_text)
                    current_length += element_length
            
            # Process final chunk if any
            if current_chunk:
                chunk_html = "".join(current_chunk)
                try:
                    # Use the same prompt format for consistency
                    prompt = f"""Convert the following HTML to clean, well-formatted Markdown.
Follow these rules:
1. Preserve headings, lists, links, and basic formatting
2. Remove any unnecessary HTML attributes or styling
3. Keep the document structure intact
4. Use proper Markdown syntax for links, images, and formatting

HTML Content:
{chunk_html}

Convert the above HTML to Markdown format."""

                    response = self.llm.generate_text(prompt)
                    chunks.append(response["text"].strip())
                except Exception as e:
                    logger.warning(f"Error converting final chunk: {str(e)}")
                    # Try to process the chunk with simpler conversion
                    from html2text import HTML2Text
                    h = HTML2Text()
                    h.ignore_links = False
                    chunks.append(h.handle(chunk_html))
            
            # Combine all chunks
            markdown_text = "\n\n".join(chunks)
            
            # Calculate word count and reading time
            word_count = len(markdown_text.split())
            reading_time = max(1, word_count // 200)  # Assume 200 words per minute
            
            # Create conversion result
            result = ConversionResult(
                markdown=markdown_text,
                author=metadata.get("author"),
                publication_date=metadata.get("date"),
                summary=None,  # We'll generate this if needed
                tags=[],  # We'll extract these if needed
                word_count=word_count,
                reading_time=reading_time
            )
            
            # Generate summary if content is long enough
            if word_count > self.summary_threshold:
                summary_prompt = f"""Generate a concise summary (max {self.max_summary_length} words) of the following text. Focus on the main points and key information:

{markdown_text[:2000]}"""

                try:
                    summary_response = self.llm.generate_text(summary_prompt)
                    result.summary = summary_response["text"].strip()
                except Exception as e:
                    logger.warning(f"Failed to generate summary: {str(e)}")
            
            return result.to_dict()
            
        except Exception as e:
            error_msg = f"Failed to convert HTML to Markdown: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

def create_converter(model_name: str = "gemma2-9b-it") -> HTMLToMarkdownConverter:
    """
    Factory function to create a converter instance with default settings.

    Args:
        model_name: Name of the Groq model to use

    Returns:
        Configured HTMLToMarkdownConverter instance
    """
    return HTMLToMarkdownConverter(
        model_name=model_name,
        summary_threshold=500,
        max_summary_length=150
    ) 