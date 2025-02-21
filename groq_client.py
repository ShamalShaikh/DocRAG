"""
Groq API Client Module.

This module provides a unified interface for interacting with Groq's API
for both text generation and HTML-to-Markdown conversion tasks.
"""

import logging
import os
from typing import Dict, Any, Optional

from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GroqClient:
    """
    A client for interacting with Groq's API.
    
    This class provides a unified interface for:
    1. Text generation for QA
    2. HTML to Markdown conversion
    3. Other LLM-based tasks
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemma2-9b-it",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs  # Accept but ignore additional kwargs
    ):
        """
        Initialize the Groq client.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model_name: Name of the model to use
            max_tokens: Maximum tokens for responses
            temperature: Temperature for response generation
            **kwargs: Additional arguments (ignored)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not provided. Set GROQ_API_KEY environment "
                "variable or pass api_key to constructor."
            )
            
        # Initialize Groq client with only the required parameters
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info(f"Initialized Groq client with model: {model_name}")
        
    def generate_text(self, prompt: str) -> Dict[str, Any]:
        """
        Generate text using Groq's API.
        
        Args:
            prompt: Input prompt for text generation
            
        Returns:
            Dict containing response text and token usage
            
        Raises:
            Exception: If the API call fails
        """
        try:
            # Format the message for the chat completion API
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides clear, accurate, and well-formatted responses."},
                {"role": "user", "content": prompt}
            ]
            
            # Make API call with proper parameters
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            result = {
                "text": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating text with Groq: {str(e)}")
            raise
            
    def convert_html_to_markdown(self, html: str) -> Dict[str, Any]:
        """
        Convert HTML to Markdown using Groq's API.
        
        Args:
            html: HTML content to convert
            
        Returns:
            Dict containing Markdown text and metadata
            
        Raises:
            Exception: If the API call fails
        """
        prompt = f"""Convert the following HTML to clean, well-formatted Markdown.
Preserve the document structure and formatting.
Extract any available metadata (title, author, date, etc.).

HTML Content:
{html}

Output the conversion as a JSON object with these fields:
- markdown: The converted Markdown text
- metadata: Any extracted metadata (title, author, date)
"""
        
        try:
            result = self.generate_text(prompt)
            # Parse the JSON response from the LLM
            # The response should be in the format:
            # {
            #   "markdown": "# Title\n\nContent...",
            #   "metadata": {"title": "...", "author": "...", "date": "..."}
            # }
            return result
            
        except Exception as e:
            logger.error(f"Error converting HTML to Markdown: {str(e)}")
            raise

def create_groq_client(**kwargs) -> GroqClient:
    """
    Factory function to create a GroqClient instance.
    
    Args:
        **kwargs: Arguments to pass to GroqClient constructor
        
    Returns:
        Configured GroqClient instance
    """
    # Filter out any unexpected kwargs before passing to GroqClient
    valid_params = ['api_key', 'model_name', 'max_tokens', 'temperature']
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return GroqClient(**filtered_kwargs) 