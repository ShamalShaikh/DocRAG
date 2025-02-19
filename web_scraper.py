"""
Web Scraping Module for Data Ingestion System.

This module provides a robust web scraping solution that supports both API-based
and HTML-based data extraction, with features including multi-threading,
scheduled scraping, and comprehensive error handling.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingConfig:
    """Configuration class for web scraping parameters."""
    use_api: bool = False
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    headers: Dict[str, str] = None
    timeout: int = 30
    max_retries: int = 3
    max_workers: int = 5
    scrape_interval: int = 3600  # Default: 1 hour
    
    def __post_init__(self):
        """Set default headers if none provided."""
        if self.headers is None:
            self.headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

class WebScraper:
    """
    A versatile web scraping class that supports both API and HTML scraping methods.
    
    Features:
    - API and HTML scraping capabilities
    - Multi-threaded scraping
    - Scheduled scraping
    - Robust error handling
    - Configurable retry mechanism
    
    Attributes:
        config (ScrapingConfig): Configuration parameters for the scraper
        scheduler (BackgroundScheduler): Scheduler for periodic scraping tasks
    """
    
    def __init__(self, config: ScrapingConfig):
        """
        Initialize the WebScraper with the given configuration.
        
        Args:
            config (ScrapingConfig): Configuration parameters for the scraper
        """
        self.config = config
        self.scheduler = BackgroundScheduler()
        self.session = requests.Session()
        
        # Add retry mechanism to the session
        adapter = requests.adapters.HTTPAdapter(max_retries=config.max_retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
    def fetch_data_from_api(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Fetch data from an API endpoint.
        
        Args:
            endpoint (str): API endpoint to fetch data from
            params (Dict, optional): Query parameters for the API request
            
        Returns:
            Dict: JSON response from the API
            
        Raises:
            RequestException: If the API request fails
        """
        if not self.config.api_base_url:
            raise ValueError("API base URL not configured")
            
        url = urljoin(self.config.api_base_url, endpoint)
        headers = self.config.headers.copy()
        
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
            
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
            
    def scrape_html(self, url: str, parser: Callable[[BeautifulSoup], Any]) -> Any:
        """
        Scrape data from an HTML page using BeautifulSoup.
        
        Args:
            url (str): URL to scrape
            parser (Callable): Function to parse the BeautifulSoup object
            
        Returns:
            Any: Parsed data from the HTML
            
        Raises:
            RequestException: If the HTTP request fails
            ValueError: If the parser function fails
        """
        try:
            response = self.session.get(
                url,
                headers=self.config.headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return parser(soup)
            
        except RequestException as e:
            logger.error(f"Failed to fetch HTML from {url}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse HTML from {url}: {str(e)}")
            raise ValueError(f"HTML parsing failed: {str(e)}")
            
    def scrape_urls_parallel(
        self,
        urls: List[str],
        parser: Callable[[BeautifulSoup], Any]
    ) -> Dict[str, Any]:
        """
        Scrape multiple URLs in parallel using multi-threading.
        
        Args:
            urls (List[str]): List of URLs to scrape
            parser (Callable): Function to parse the BeautifulSoup object
            
        Returns:
            Dict[str, Any]: Dictionary mapping URLs to their scraped data
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_url = {
                executor.submit(self.scrape_html, url, parser): url
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                except Exception as e:
                    logger.error(f"Error scraping {url}: {str(e)}")
                    results[url] = None
                    
        return results
        
    def schedule_scraping(
        self,
        task: Callable,
        interval: Optional[int] = None,
        **task_kwargs
    ) -> None:
        """
        Schedule a scraping task to run periodically.
        
        Args:
            task (Callable): The scraping task to schedule
            interval (int, optional): Interval in seconds (defaults to config value)
            **task_kwargs: Additional arguments to pass to the task
        """
        if not interval:
            interval = self.config.scrape_interval
            
        self.scheduler.add_job(
            task,
            'interval',
            seconds=interval,
            kwargs=task_kwargs,
            next_run_time=datetime.now()  # Run immediately first
        )
        
        if not self.scheduler.running:
            self.scheduler.start()
            
    def stop_scheduler(self) -> None:
        """Stop the scheduler if it's running."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop_scheduler()
        self.session.close()

# Example usage and helper functions
def example_html_parser(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Example parser function to extract data from HTML.
    Customize this based on your specific needs.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        Dict[str, str]: Extracted data
    """
    return {
        'title': soup.title.string if soup.title else None,
        'h1_headers': [h1.text for h1 in soup.find_all('h1')],
        'meta_description': soup.find('meta', {'name': 'description'})['content']
        if soup.find('meta', {'name': 'description'})
        else None
    }

def create_default_scraper(
    use_api: bool = False,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None
) -> WebScraper:
    """
    Create a WebScraper instance with default configuration.
    
    Args:
        use_api (bool): Whether to use API-based scraping
        api_key (str, optional): API key for authentication
        api_base_url (str, optional): Base URL for API endpoints
        
    Returns:
        WebScraper: Configured WebScraper instance
    """
    config = ScrapingConfig(
        use_api=use_api,
        api_key=api_key,
        api_base_url=api_base_url,
        max_workers=5,
        scrape_interval=3600
    )
    return WebScraper(config)

if __name__ == "__main__":
    # Example usage
    urls_to_scrape = [
        "https://example.com",
        "https://example.org"
    ]
    
    scraper = create_default_scraper()
    
    # Example of parallel scraping
    results = scraper.scrape_urls_parallel(urls_to_scrape, example_html_parser)
    
    # Example of scheduled scraping
    def scheduled_task(urls):
        print(f"Running scheduled scrape at {datetime.now()}")
        results = scraper.scrape_urls_parallel(urls, example_html_parser)
        print(f"Scraped {len(results)} URLs")
        
    # Schedule the task to run every hour
    scraper.schedule_scraping(scheduled_task, urls=urls_to_scrape)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scraper.stop_scheduler() 