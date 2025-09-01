"""
Web scraper for RAG system.
Scrapes websites and converts them into documents for indexing.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse
from typing import List, Dict, Any, Optional, Set
from loguru import logger
import time
import re
from pathlib import Path


class WebScraper:
    """Scrape websites and extract content for RAG system."""
    
    def __init__(self, 
                 delay: float = 1.0,
                 max_pages: int = 100,
                 respect_robots: bool = True):
        """
        Initialize web scraper.
        
        Args:
            delay: Delay between requests (seconds)
            max_pages: Maximum pages to scrape per domain
            respect_robots: Whether to respect robots.txt
        """
        self.delay = delay
        self.max_pages = max_pages
        self.respect_robots = respect_robots
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent comparison."""
        try:
            parsed = urlparse(url.lower())
            # Remove trailing slash, fragment, and common query params
            path = parsed.path.rstrip('/')
            # Keep essential query params, remove tracking ones
            query = parsed.query
            if query:
                # Remove common tracking parameters
                tracking_params = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 
                                 'fbclid', 'gclid', '_ga', '_gl', 'ref', 'source'}
                query_dict = parse_qs(query)
                filtered_query = {k: v for k, v in query_dict.items() 
                                if k not in tracking_params}
                # Sort for consistency
                query = '&'.join(f"{k}={v[0]}" for k, v in sorted(filtered_query.items()))
            
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc,
                path,
                parsed.params,
                query,
                ''  # Remove fragment
            ))
            return normalized
        except Exception:
            return url.lower().rstrip('/')
        
    def scrape_website(self, 
                      base_url: str, 
                      max_depth: int = 2,
                      include_patterns: Optional[List[str]] = None,
                      exclude_patterns: Optional[List[str]] = None,
                      existing_urls: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Scrape a website and return documents.
        
        Args:
            base_url: Base URL to start scraping
            max_depth: Maximum depth to crawl
            include_patterns: URL patterns to include
            exclude_patterns: URL patterns to exclude
            existing_urls: URLs that have already been scraped (for deduplication)
            
        Returns:
            List of document dictionaries
        """
        logger.info(f"Starting to scrape website: {base_url}")
        
        # Initialize visited URLs for this session
        self.visited_urls = set()
        
        # Normalize existing URLs for comparison
        normalized_existing = set()
        if existing_urls:
            for url in existing_urls:
                normalized = self._normalize_url(url)
                normalized_existing.add(normalized)
            logger.info(f"Will skip content from {len(existing_urls)} already scraped URLs, but still discover links")
            logger.debug(f"First 5 normalized existing URLs: {list(normalized_existing)[:5]}")
        
        # Clean and validate URL
        base_url = self._clean_url(base_url)
        if not self._is_valid_url(base_url):
            logger.error(f"Invalid URL: {base_url}")
            return []
        
        documents = []
        urls_to_process = [(base_url, 0)]  # (url, depth)
        
        while urls_to_process and len(documents) < self.max_pages:
            current_url, depth = urls_to_process.pop(0)
            
            # Skip if too deep
            if depth > max_depth:
                continue
            
            # Check URL patterns
            if not self._should_scrape_url(current_url, include_patterns, exclude_patterns):
                continue
            
            # Skip if we've already processed this URL in this session
            if current_url in self.visited_urls:
                continue
                
            try:
                # Check if this URL was already scraped in previous sessions (normalized comparison)
                normalized_current = self._normalize_url(current_url)
                already_scraped = normalized_current in normalized_existing
                
                if already_scraped:
                    logger.debug(f"Skipping already scraped: {current_url} (normalized: {normalized_current})")
                
                if not already_scraped:
                    # Scrape the page for new content
                    doc = self._scrape_page(current_url)
                    if doc:
                        documents.append(doc)
                        logger.info(f"Scraped: {current_url} ({len(documents)}/{self.max_pages})")
                else:
                    logger.debug(f"Skipping already scraped: {current_url}")
                
                # Always extract links for discovery (even from already-scraped pages)
                if depth < max_depth:
                    new_urls = self._extract_links(current_url, base_url)
                    for new_url in new_urls:
                        if new_url not in self.visited_urls:
                            urls_to_process.append((new_url, depth + 1))
                
                # Mark as visited in this session
                self.visited_urls.add(current_url)
                
                # Be respectful - add delay
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error scraping {current_url}: {e}")
                continue
        
        logger.info(f"Completed scraping. Found {len(documents)} documents from {base_url}")
        return documents
    
    def _scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single page and extract content."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Skip if content is too short
            if len(content.strip()) < 100:
                logger.warning(f"Content too short for {url}")
                return None
            
            # Extract metadata
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content', '') if meta_description else ''
            
            # Create document
            document = {
                'content': content,
                'processed_content': content,  # Will be processed later
                'title': title_text,
                'url': url,
                'filename': self._url_to_filename(url),
                'file_type': '.html',
                'file_path': url,
                'description': description,
                'word_count': len(content.split()),
                'char_count': len(content),
                'source_type': 'website',
                'scraped_at': time.time()
            }
            
            return document
            
        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML soup."""
        # Try to find main content areas
        main_selectors = [
            'main',
            'article', 
            '.content',
            '.main-content',
            '#content',
            '#main',
            '.post-content',
            '.entry-content',
            '.article-content'
        ]
        
        # Try each selector
        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text() for elem in elements])
                if len(content.strip()) > 200:  # Good content length
                    return self._clean_text(content)
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            content = body.get_text()
            return self._clean_text(content)
        
        # Last resort - all text
        return self._clean_text(soup.get_text())
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common unwanted patterns
        text = re.sub(r'Cookie.*?Accept', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Privacy Policy', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Terms of Service', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_links(self, current_url: str, base_url: str) -> List[str]:
        """Extract links from a page."""
        try:
            response = self.session.get(current_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(current_url, href)
                
                # Only include links from the same domain
                if self._is_same_domain(full_url, base_url):
                    links.append(self._clean_url(full_url))
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting links from {current_url}: {e}")
            return []
    
    def _clean_url(self, url: str) -> str:
        """Clean and normalize URL."""
        # Remove fragments and some query parameters
        parsed = urlparse(url)
        
        # Remove common tracking parameters
        query_params = parse_qs(parsed.query)
        clean_params = {k: v for k, v in query_params.items() 
                       if k not in ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content']}
        
        # Rebuild URL
        from urllib.parse import urlencode, urlunparse
        clean_query = urlencode(clean_params, doseq=True)
        
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            clean_query,
            ''  # Remove fragment
        ))
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except:
            return False
    
    def _is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain."""
        try:
            domain1 = urlparse(url1).netloc.lower()
            domain2 = urlparse(url2).netloc.lower()
            return domain1 == domain2
        except:
            return False
    
    def _should_scrape_url(self, url: str, 
                          include_patterns: Optional[List[str]], 
                          exclude_patterns: Optional[List[str]]) -> bool:
        """Check if URL should be scraped based on patterns."""
        # Check exclude patterns first
        if exclude_patterns:
            for pattern in exclude_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False
        
        # Check include patterns
        if include_patterns:
            for pattern in include_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return True
            return False  # If include patterns specified but none match
        
        # Default: scrape if no patterns or passed exclude check
        return True
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to a filename."""
        parsed = urlparse(url)
        
        # Create filename from path
        path = parsed.path.strip('/')
        if not path:
            path = 'index'
        
        # Replace invalid filename characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', path)
        filename = re.sub(r'[/\\]', '_', filename)
        
        # Add domain prefix
        domain = parsed.netloc.replace('www.', '')
        
        return f"{domain}_{filename}.html"
    
    def scrape_multiple_websites(self, 
                                urls: List[str],
                                max_depth: int = 2,
                                include_patterns: Optional[List[str]] = None,
                                exclude_patterns: Optional[List[str]] = None,
                                existing_urls: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Scrape multiple websites.
        
        Args:
            urls: List of base URLs to scrape
            max_depth: Maximum depth to crawl for each site
            include_patterns: URL patterns to include
            exclude_patterns: URL patterns to exclude
            existing_urls: URLs that have already been scraped (for deduplication)
            
        Returns:
            Combined list of documents from all websites
        """
        all_documents = []
        
        for url in urls:
            logger.info(f"Scraping website {len(all_documents)+1}/{len(urls)}: {url}")
            
            try:
                documents = self.scrape_website(
                    base_url=url,
                    max_depth=max_depth,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    existing_urls=existing_urls
                )
                all_documents.extend(documents)
                
                # Note: Keep visited URLs to avoid duplicates across websites
                
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                continue
        
        logger.info(f"Completed scraping {len(urls)} websites. Total documents: {len(all_documents)}")
        return all_documents
