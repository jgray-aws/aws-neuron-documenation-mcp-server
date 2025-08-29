#!/usr/bin/env python3
"""
Test script for the AWS Neuron Documentation MCP Server
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_server():
    """Test the server functionality."""
    print("🚀 Testing AWS Neuron Documentation MCP Server with ChromaDB")
    print("=" * 60)
    
    # Import server module
    import aws_neuron_documentation_mcp_server.server as server_module
    
    # Initialize database
    print("\n1. Initializing database...")
    await server_module.initialize_db()
    print("✅ Database initialized")
    
    # Check initial stats
    print("\n2. Checking database stats...")
    if server_module.collection:
        count = server_module.collection.count()
        print(f"Database has {count} documents")
    else:
        print("Collection not initialized")
        return
    
    # Index documentation if needed
    if count == 0:
        print("\n3. Indexing documentation...")
        await server_module.crawl_and_index_docs()
        final_count = server_module.collection.count()
        print(f"✅ Successfully indexed {final_count} documents")
    else:
        print(f"\n3. Database already has {count} documents")
    
    # Test search
    print("\n4. Testing search functionality...")
    search_queries = [
        "PyTorch installation",
        "neuron compiler", 
        "inference optimization"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Searching for: '{query}'")
        try:
            results = server_module.collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if results['documents'] and results['documents'][0]:
                print(f"Found {len(results['documents'][0])} results:")
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                ), 1):
                    title = metadata.get('title', 'Untitled')
                    url = metadata.get('url', '')
                    snippet = doc[:150] + "..." if len(doc) > 150 else doc
                    print(f"  {i}. {title}")
                    print(f"     URL: {url}")
                    print(f"     Relevance: {1 - distance:.3f}")
                    print(f"     Snippet: {snippet}")
            else:
                print("No results found")
        except Exception as e:
            print(f"Search error: {e}")
        print("-" * 40)
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_server())