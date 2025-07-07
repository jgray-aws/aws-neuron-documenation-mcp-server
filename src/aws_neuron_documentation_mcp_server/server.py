#!/usr/bin/env python3
"""
AWS Neuron Documentation MCP Server

This server provides tools to search and retrieve AWS Neuron documentation.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    ServerCapabilities
)
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aws-neuron-docs-mcp")

# AWS Neuron Documentation base URL
NEURON_DOCS_BASE_URL = "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/"

class SearchResult(BaseModel):
    """Model for search results."""
    title: str
    url: str
    snippet: str
    section: Optional[str] = None

class NeuronDocumentationServer:
    """AWS Neuron Documentation MCP Server."""
    
    def __init__(self):
        self.server = Server("aws-neuron-documentation-mcp-server")
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "AWS-Neuron-Documentation-MCP-Server/0.1.0"
            }
        )
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_neuron_docs",
                    description="Search AWS Neuron documentation for specific topics, APIs, or concepts",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for AWS Neuron documentation"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_neuron_doc_content",
                    description="Retrieve the full content of a specific AWS Neuron documentation page",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL of the AWS Neuron documentation page to retrieve"
                            }
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="list_neuron_guides",
                    description="List available AWS Neuron guides and tutorials by category",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Category to filter guides (e.g., 'getting-started', 'tutorials', 'frameworks')",
                                "enum": ["all", "getting-started", "tutorials", "frameworks", "inference", "training"]
                            }
                        },
                        "required": ["category"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "search_neuron_docs":
                    return await self._search_neuron_docs(
                        query=arguments["query"],
                        max_results=arguments.get("max_results", 10)
                    )
                elif name == "get_neuron_doc_content":
                    return await self._get_neuron_doc_content(
                        url=arguments["url"]
                    )
                elif name == "list_neuron_guides":
                    return await self._list_neuron_guides(
                        category=arguments["category"]
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]
    
    async def _search_neuron_docs(self, query: str, max_results: int = 10) -> List[TextContent]:
        """Search AWS Neuron documentation."""
        try:
            # Use the search functionality of the documentation site
            search_url = f"{NEURON_DOCS_BASE_URL}search.html"
            params = {"q": query}
            
            response = await self.http_client.get(search_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Look for search results in the page
            search_results = soup.find_all('div', class_='search-result') or soup.find_all('li', class_='search-result')
            
            if not search_results:
                # Fallback: search through the main documentation pages
                results = await self._fallback_search(query, max_results)
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
                        
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet
                        ))
            
            if not results:
                return [TextContent(
                    type="text",
                    text=f"No results found for query: {query}"
                )]
            
            # Format results
            formatted_results = []
            formatted_results.append(f"Found {len(results)} results for '{query}':\n")
            
            for i, result in enumerate(results, 1):
                formatted_results.append(f"{i}. **{result.title}**")
                formatted_results.append(f"   URL: {result.url}")
                if result.snippet:
                    formatted_results.append(f"   {result.snippet}")
                formatted_results.append("")
            
            return [TextContent(
                type="text",
                text="\n".join(formatted_results)
            )]
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return [TextContent(
                type="text",
                text=f"Error searching documentation: {str(e)}"
            )]
    
    async def _fallback_search(self, query: str, max_results: int) -> List[SearchResult]:
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
                response = await self.http_client.get(section_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for links and content that match the query
                    links = soup.find_all('a', href=True)
                    for link in links:
                        link_text = link.get_text(strip=True).lower()
                        if query_lower in link_text and len(results) < max_results:
                            url = urljoin(section_url, link['href'])
                            results.append(SearchResult(
                                title=link.get_text(strip=True),
                                url=url,
                                snippet=f"Found in {section} section",
                                section=section
                            ))
                
            except Exception as e:
                logger.warning(f"Error searching section {section}: {str(e)}")
                continue
        
        return results
    
    async def _get_neuron_doc_content(self, url: str) -> List[TextContent]:
        """Retrieve content from a specific documentation page."""
        try:
            # Ensure URL is valid and from the Neuron docs domain
            parsed_url = urlparse(url)
            if not parsed_url.netloc and not url.startswith(NEURON_DOCS_BASE_URL):
                url = urljoin(NEURON_DOCS_BASE_URL, url)
            
            response = await self.http_client.get(url)
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
                
                return [TextContent(
                    type="text",
                    text=f"# {title_text}\n\nSource: {url}\n\n{clean_content}"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Could not extract content from {url}"
                )]
                
        except Exception as e:
            logger.error(f"Error retrieving content from {url}: {str(e)}")
            return [TextContent(
                type="text",
                text=f"Error retrieving content: {str(e)}"
            )]
    
    async def _list_neuron_guides(self, category: str = "all") -> List[TextContent]:
        """List available AWS Neuron guides by category."""
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
            
            return [TextContent(
                type="text",
                text="\n".join(result_lines)
            )]
            
        except Exception as e:
            logger.error(f"Error listing guides: {str(e)}")
            return [TextContent(
                type="text",
                text=f"Error listing guides: {str(e)}"
            )]
    
    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="aws-neuron-documentation-mcp-server",
                    server_version="0.1.0",
                    capabilities=ServerCapabilities()
                ),
            )

async def main():
    """Main entry point."""
    server = NeuronDocumentationServer()
    await server.run()

def run_server():
    """Synchronous wrapper for the async main function."""
    asyncio.run(main())

if __name__ == "__main__":
    run_server()
