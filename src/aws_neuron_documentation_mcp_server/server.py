#!/usr/bin/env python3
"""
AWS Neuron Documentation MCP Server using FastMCP

This server provides tools to search and retrieve AWS Neuron documentation.
"""

import logging
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aws-neuron-docs-mcp")

# AWS Neuron Documentation base URL
NEURON_DOCS_BASE_URL = "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/"

# Initialize FastMCP
mcp = FastMCP("AWS Neuron Documentation")

# HTTP client for making requests
http_client = httpx.AsyncClient(
    timeout=30.0,
    follow_redirects=True,
    headers={
        "User-Agent": "AWS-Neuron-Documentation-MCP-Server/0.1.0"
    }
)

@mcp.tool()
async def search_neuron_docs(query: str, max_results: int = 10) -> str:
    """Search AWS Neuron documentation for specific topics, APIs, or concepts.
    
    Args:
        query: Search query for AWS Neuron documentation
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    try:
        # Use the search functionality of the documentation site
        search_url = f"{NEURON_DOCS_BASE_URL}search.html"
        params = {"q": query}
        
        response = await http_client.get(search_url, params=params)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Look for search results in the page
        search_results = soup.find_all('div', class_='search-result') or soup.find_all('li', class_='search-result')
        
        if not search_results:
            # Fallback: search through the main documentation pages
            results = await _fallback_search(query, max_results)
        else:
            for result in search_results[:max_results]:
                title_elem = result.find('a') or result.find('h3')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    if url and not url.startswith('http'):
                        url = urljoin(NEURON_DOCS_BASE_URL, url)
                    
                    snippet_elem = result.find('p') or result.find('div', class_='snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet
                    })
        
        if not results:
            return f"No results found for query: {query}"
        
        # Format results
        formatted_results = [f"Found {len(results)} results for '{query}':\n"]
        
        for i, result in enumerate(results, 1):
            formatted_results.append(f"{i}. **{result['title']}**")
            formatted_results.append(f"   URL: {result['url']}")
            if result['snippet']:
                formatted_results.append(f"   {result['snippet']}")
            formatted_results.append("")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"Error searching documentation: {str(e)}"

@mcp.tool()
async def get_neuron_doc_content(url: str) -> str:
    """Retrieve the full content of a specific AWS Neuron documentation page.
    
    Args:
        url: URL of the AWS Neuron documentation page to retrieve
    
    Returns:
        Full text content of the documentation page
    """
    try:
        # Ensure URL is valid and from the Neuron docs domain
        parsed_url = urlparse(url)
        if not parsed_url.netloc and not url.startswith(NEURON_DOCS_BASE_URL):
            url = urljoin(NEURON_DOCS_BASE_URL, url)
        
        response = await http_client.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content
        main_content = soup.find('div', class_='document') or soup.find('main') or soup.find('article')
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Remove navigation, sidebar, and footer elements
            for elem in main_content.find_all(['nav', 'aside', 'footer', 'script', 'style']):
                elem.decompose()
            
            # Get title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else "AWS Neuron Documentation"
            
            # Get clean text content
            content = main_content.get_text(separator='\n', strip=True)
            
            # Clean up excessive whitespace
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            clean_content = '\n'.join(lines)
            
            return f"# {title_text}\n\nSource: {url}\n\n{clean_content}"
        else:
            return f"Could not extract content from {url}"
            
    except Exception as e:
        logger.error(f"Error retrieving content from {url}: {str(e)}")
        return f"Error retrieving content: {str(e)}"

@mcp.tool()
async def list_neuron_guides(category: str) -> str:
    """List available AWS Neuron guides and tutorials by category.
    
    Args:
        category: Category to filter guides. Options: 'all', 'getting-started', 
                 'tutorials', 'frameworks', 'inference', 'training'
    
    Returns:
        Formatted list of guides with titles and URLs
    """
    try:
        guides = {
            "getting-started": [
                ("Quick Start Guide", "neuron-guide/neuron-quick-start/"),
                ("Installation Guide", "neuron-guide/neuron-install-guide/"),
                ("Setup Guide", "neuron-guide/neuron-setup/")
            ],
            "tutorials": [
                ("PyTorch Tutorial", "frameworks/torch/torch-neuron/"),
                ("TensorFlow Tutorial", "frameworks/tensorflow/tensorflow-neuron/"),
                ("MXNet Tutorial", "frameworks/mxnet/mxnet-neuron/")
            ],
            "frameworks": [
                ("PyTorch Neuron", "frameworks/torch/"),
                ("TensorFlow Neuron", "frameworks/tensorflow/"),
                ("MXNet Neuron", "frameworks/mxnet/")
            ],
            "inference": [
                ("Neuron Runtime", "neuron-runtime/"),
                ("Model Serving", "neuron-guide/neuron-model-serving/"),
                ("Performance Optimization", "neuron-guide/performance/")
            ],
            "training": [
                ("Distributed Training", "neuron-guide/training/"),
                ("Training Best Practices", "neuron-guide/training-best-practices/")
            ]
        }
        
        result_lines = []
        
        if category == "all":
            result_lines.append("# AWS Neuron Documentation Guides\n")
            for cat, guide_list in guides.items():
                result_lines.append(f"## {cat.replace('-', ' ').title()}")
                for title, path in guide_list:
                    url = urljoin(NEURON_DOCS_BASE_URL, path)
                    result_lines.append(f"- **{title}**: {url}")
                result_lines.append("")
        else:
            if category in guides:
                result_lines.append(f"# {category.replace('-', ' ').title()} Guides\n")
                for title, path in guides[category]:
                    url = urljoin(NEURON_DOCS_BASE_URL, path)
                    result_lines.append(f"- **{title}**: {url}")
            else:
                result_lines.append(f"Category '{category}' not found. Available categories: {', '.join(guides.keys())}")
        
        return "\n".join(result_lines)
        
    except Exception as e:
        logger.error(f"Error listing guides: {str(e)}")
        return f"Error listing guides: {str(e)}"

async def _fallback_search(query: str, max_results: int) -> List[dict]:
    """Fallback search by crawling main documentation sections."""
    results = []
    query_lower = query.lower()
    
    # Common AWS Neuron documentation sections
    sections = [
        "neuron-guide/",
        "frameworks/",
        "neuron-runtime/",
        "neuron-compiler/",
        "tools/",
        "release-notes/"
    ]
    
    for section in sections:
        try:
            section_url = urljoin(NEURON_DOCS_BASE_URL, section)
            response = await http_client.get(section_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for links and content that match the query
                links = soup.find_all('a', href=True)
                for link in links:
                    link_text = link.get_text(strip=True).lower()
                    if query_lower in link_text and len(results) < max_results:
                        url = urljoin(section_url, link['href'])
                        results.append({
                            'title': link.get_text(strip=True),
                            'url': url,
                            'snippet': f"Found in {section} section"
                        })
            
        except Exception as e:
            logger.warning(f"Error searching section {section}: {str(e)}")
            continue
    
    return results

if __name__ == "__main__":
    mcp.run()