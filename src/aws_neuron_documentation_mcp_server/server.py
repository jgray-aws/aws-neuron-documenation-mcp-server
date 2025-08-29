#!/usr/bin/env python3
"""
AWS Neuron Documentation MCP Server using FastMCP and ChromaDB

This server provides tools to search and retrieve AWS Neuron documentation
using a local ChromaDB vector database for fast semantic search.
"""

import asyncio
import logging
import os
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse

import aiofiles
import chromadb
import httpx
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

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

# ChromaDB setup
DB_PATH = Path.home() / ".aws_neuron_docs_mcp" / "chroma_db"
COLLECTION_NAME = "neuron_docs"
chroma_client = None
collection = None
embedding_model = None

async def initialize_db():
    """Initialize ChromaDB and embedding model."""
    global chroma_client, collection, embedding_model
    
    try:
        # Create database directory
        DB_PATH.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
            logger.info(f"Loaded existing collection with {collection.count()} documents")
        except Exception:
            collection = chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "AWS Neuron Documentation"}
            )
            logger.info("Created new collection")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

async def crawl_and_index_docs():
    """Crawl AWS Neuron documentation and index it in ChromaDB."""
    if not collection:
        await initialize_db()
    
    logger.info("Starting documentation crawl and indexing...")
    
    # Common documentation sections to crawl
    sections_to_crawl = [
        "",  # Main page
        "neuron-guide/",
        "frameworks/",
        "neuron-runtime/",
        "neuron-compiler/", 
        "tools/",
        "release-notes/",
        "neuron-guide/neuron-quick-start/",
        "neuron-guide/neuron-install-guide/",
        "frameworks/torch/",
        "frameworks/tensorflow/",
        "frameworks/mxnet/"
    ]
    
    indexed_urls = set()
    documents = []
    metadatas = []
    ids = []
    
    for section in sections_to_crawl:
        try:
            section_url = urljoin(NEURON_DOCS_BASE_URL, section)
            await _crawl_section(section_url, indexed_urls, documents, metadatas, ids)
        except Exception as e:
            logger.warning(f"Error crawling section {section}: {e}")
    
    if documents:
        # Generate embeddings and add to collection
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Split into batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            try:
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
        
        logger.info(f"Successfully indexed {len(documents)} documents")
    else:
        logger.warning("No documents found to index")

async def _crawl_section(url: str, indexed_urls: set, documents: list, metadatas: list, ids: list):
    """Crawl a documentation section and extract content."""
    if url in indexed_urls:
        return
    
    indexed_urls.add(url)
    
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content
        main_content = soup.find('div', class_='document') or soup.find('main') or soup.find('article')
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Remove navigation, sidebar, and footer elements
            for elem in main_content.find_all(['nav', 'aside', 'footer', 'script', 'style', 'header']):
                elem.decompose()
            
            # Get title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else "AWS Neuron Documentation"
            
            # Get clean text content
            content = main_content.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            clean_content = '\n'.join(lines)
            
            if clean_content and len(clean_content) > 100:  # Only index substantial content
                doc_id = hashlib.md5(url.encode()).hexdigest()
                
                documents.append(clean_content)
                metadatas.append({
                    "title": title_text,
                    "url": url,
                    "section": _get_section_from_url(url)
                })
                ids.append(doc_id)
                
                logger.debug(f"Indexed: {title_text}")
        
        # Find and crawl linked pages within the same domain
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            if href.startswith('/') or href.startswith('./') or (not href.startswith('http')):
                full_url = urljoin(url, href)
                if full_url.startswith(NEURON_DOCS_BASE_URL) and full_url not in indexed_urls:
                    # Limit depth to avoid infinite crawling
                    if len(indexed_urls) < 200:  # Reasonable limit
                        await _crawl_section(full_url, indexed_urls, documents, metadatas, ids)
                        
    except Exception as e:
        logger.warning(f"Error crawling {url}: {e}")

def _get_section_from_url(url: str) -> str:
    """Extract section name from URL."""
    path = urlparse(url).path
    if '/neuron-guide/' in path:
        return 'neuron-guide'
    elif '/frameworks/' in path:
        return 'frameworks'
    elif '/neuron-runtime/' in path:
        return 'neuron-runtime'
    elif '/neuron-compiler/' in path:
        return 'neuron-compiler'
    elif '/tools/' in path:
        return 'tools'
    elif '/release-notes/' in path:
        return 'release-notes'
    else:
        return 'general'

@mcp.tool()
async def index_neuron_docs() -> str:
    """Index AWS Neuron documentation into local ChromaDB for fast searching.
    
    Returns:
        Status message about the indexing process
    """
    try:
        if not collection:
            await initialize_db()
        
        # Check if already indexed
        count = collection.count()
        if count > 0:
            return f"Documentation already indexed with {count} documents. Use 'reindex_neuron_docs' to refresh."
        
        await crawl_and_index_docs()
        final_count = collection.count()
        return f"Successfully indexed {final_count} AWS Neuron documentation pages into local database."
        
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        return f"Error indexing documentation: {e}"

@mcp.tool()
async def reindex_neuron_docs() -> str:
    """Re-index AWS Neuron documentation, replacing existing data.
    
    Returns:
        Status message about the re-indexing process
    """
    global collection
    try:
        if not collection:
            await initialize_db()
        
        # Clear existing collection
        chroma_client.delete_collection(COLLECTION_NAME)
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "AWS Neuron Documentation"}
        )
        
        await crawl_and_index_docs()
        final_count = collection.count()
        return f"Successfully re-indexed {final_count} AWS Neuron documentation pages."
        
    except Exception as e:
        logger.error(f"Re-indexing error: {e}")
        return f"Error re-indexing documentation: {e}"

@mcp.tool()
async def search_neuron_docs(query: str, max_results: int = 10) -> str:
    """Search AWS Neuron documentation for specific topics, APIs, or concepts using semantic search.
    
    Args:
        query: Search query for AWS Neuron documentation
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    try:
        if not collection:
            await initialize_db()
        
        # Check if database is populated
        count = collection.count()
        if count == 0:
            return "Documentation not indexed yet. Please run 'index_neuron_docs' first to build the local database."
        
        # Perform semantic search
        results = collection.query(
            query_texts=[query],
            n_results=min(max_results, count)
        )
        
        if not results['documents'] or not results['documents'][0]:
            return f"No results found for query: {query}"
        
        # Format results
        formatted_results = [f"Found {len(results['documents'][0])} results for '{query}' (from {count} indexed documents):\n"]
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        ), 1):
            title = metadata.get('title', 'Untitled')
            url = metadata.get('url', '')
            section = metadata.get('section', 'general')
            
            # Create snippet from document content
            snippet = doc[:200] + "..." if len(doc) > 200 else doc
            
            formatted_results.append(f"{i}. **{title}** [{section}]")
            formatted_results.append(f"   URL: {url}")
            formatted_results.append(f"   Relevance: {1 - distance:.3f}")
            formatted_results.append(f"   {snippet}")
            formatted_results.append("")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Error searching documentation: {e}"

@mcp.tool()
async def get_neuron_doc_content(url: str) -> str:
    """Retrieve the full content of a specific AWS Neuron documentation page from local database or web.
    
    Args:
        url: URL of the AWS Neuron documentation page to retrieve
    
    Returns:
        Full text content of the documentation page
    """
    try:
        if not collection:
            await initialize_db()
        
        # First try to get from local database
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        try:
            results = collection.get(ids=[doc_id])
            if results['documents'] and results['documents'][0]:
                metadata = results['metadatas'][0]
                content = results['documents'][0]
                title = metadata.get('title', 'AWS Neuron Documentation')
                return f"# {title}\n\nSource: {url}\n\n{content}"
        except Exception:
            pass  # Fall back to web retrieval
        
        # Fallback to web retrieval if not in database
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
        logger.error(f"Error retrieving content from {url}: {e}")
        return f"Error retrieving content: {e}"

@mcp.tool()
async def get_db_stats() -> str:
    """Get statistics about the local documentation database.
    
    Returns:
        Database statistics including document count and sections
    """
    try:
        if not collection:
            await initialize_db()
        
        count = collection.count()
        if count == 0:
            return "Database is empty. Run 'index_neuron_docs' to populate it."
        
        # Get all documents to analyze sections
        all_docs = collection.get()
        sections = {}
        
        for metadata in all_docs['metadatas']:
            section = metadata.get('section', 'unknown')
            sections[section] = sections.get(section, 0) + 1
        
        result = [f"Database Statistics:\n"]
        result.append(f"Total documents: {count}")
        result.append(f"Database location: {DB_PATH}")
        result.append(f"\nDocuments by section:")
        
        for section, doc_count in sorted(sections.items()):
            result.append(f"  {section}: {doc_count} documents")
        
        return "\n".join(result)
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return f"Error getting database statistics: {e}"

@mcp.tool()
async def list_neuron_guides(category: str = "all") -> str:
    """List available AWS Neuron guides and tutorials by category from the local database.
    
    Args:
        category: Category to filter guides. Options: 'all', 'neuron-guide', 
                 'frameworks', 'neuron-runtime', 'neuron-compiler', 'tools', 'release-notes'
    
    Returns:
        Formatted list of guides with titles and URLs from the indexed documentation
    """
    try:
        if not collection:
            await initialize_db()
        
        count = collection.count()
        if count == 0:
            return "Database is empty. Run 'index_neuron_docs' first to populate the local database."
        
        # Get all documents
        all_docs = collection.get()
        
        # Group by section
        sections = {}
        for i, metadata in enumerate(all_docs['metadatas']):
            section = metadata.get('section', 'general')
            if section not in sections:
                sections[section] = []
            sections[section].append({
                'title': metadata.get('title', 'Untitled'),
                'url': metadata.get('url', ''),
                'section': section
            })
        
        result_lines = []
        
        if category == "all":
            result_lines.append("# AWS Neuron Documentation (from local database)\n")
            for section_name, docs in sorted(sections.items()):
                result_lines.append(f"## {section_name.replace('-', ' ').title()} ({len(docs)} documents)")
                for doc in sorted(docs, key=lambda x: x['title'])[:10]:  # Limit to 10 per section
                    result_lines.append(f"- **{doc['title']}**: {doc['url']}")
                if len(docs) > 10:
                    result_lines.append(f"  ... and {len(docs) - 10} more documents")
                result_lines.append("")
        else:
            if category in sections:
                docs = sections[category]
                result_lines.append(f"# {category.replace('-', ' ').title()} Documents ({len(docs)} total)\n")
                for doc in sorted(docs, key=lambda x: x['title']):
                    result_lines.append(f"- **{doc['title']}**: {doc['url']}")
            else:
                available_sections = list(sections.keys())
                result_lines.append(f"Category '{category}' not found in database.")
                result_lines.append(f"Available categories: {', '.join(available_sections)}")
        
        return "\n".join(result_lines)
        
    except Exception as e:
        logger.error(f"Error listing guides: {e}")
        return f"Error listing guides: {e}"

if __name__ == "__main__":
    # Initialize database on startup
    asyncio.create_task(initialize_db())
    mcp.run()