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
import requests
from dataclasses import dataclass
import ollama

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
                 model_name: str = "reader-lm:1.5b",
                 summary_threshold: int = 500,  # words
                 max_summary_length: int = 150):  # words
        """
        Initialize the converter with specified parameters.

        Args:
            model_name: Name of the Ollama model to use
            summary_threshold: Word count threshold for generating summaries
            max_summary_length: Maximum length of generated summaries in words
        """
        self.model_name = model_name
        self.summary_threshold = summary_threshold
        self.max_summary_length = max_summary_length
        self.llm = ollama.Client()
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
        Convert HTML content to Markdown with enhanced metadata and analysis.

        This method orchestrates the entire conversion process:
        1. Cleans the HTML
        2. Extracts metadata
        3. Converts to Markdown
        4. Generates summary if needed
        5. Generates tags
        6. Returns structured output

        Args:
            html: Raw HTML string to convert

        Returns:
            Dictionary containing converted content and metadata

        Raises:
            ValueError: If input HTML is empty or invalid
            Exception: For other conversion errors
        """
        if not html:
            raise ValueError("Input HTML cannot be empty")

        try:
            # Clean HTML and create soup object
            cleaned_html = self._clean_html(html)
            soup = BeautifulSoup(cleaned_html, 'html.parser')

            # Extract metadata
            metadata = self._extract_metadata(soup)

            # Convert to Markdown
            markdown = self._convert_to_markdown(cleaned_html)

            # Calculate word count and reading time
            word_count = len(markdown.split())
            reading_time = max(1, round(word_count / 200))  # Assuming 200 words per minute

            # Generate summary if content is long enough
            summary = self._generate_summary(markdown)

            # Generate tags
            tags = self._generate_tags(markdown, metadata)

            # Create result object
            result = ConversionResult(
                markdown=markdown,
                author=metadata['author'],
                publication_date=metadata['publication_date'],
                summary=summary,
                tags=tags,
                word_count=word_count,
                reading_time=reading_time
            )

            logger.info("Successfully completed HTML to Markdown conversion")
            return result.to_dict()

        except Exception as e:
            logger.error(f"Error during conversion process: {str(e)}")
            raise

def create_converter(model_name: str = "reader-lm:1.5b") -> HTMLToMarkdownConverter:
    """
    Factory function to create a converter instance with default settings.

    Args:
        model_name: Name of the Ollama model to use

    Returns:
        Configured HTMLToMarkdownConverter instance
    """
    return HTMLToMarkdownConverter(
        model_name=model_name,
        summary_threshold=500,
        max_summary_length=150
    ) 