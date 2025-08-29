# AWS Neuron Documentation MCP Server

An MCP (Model Context Protocol) server built with FastMCP that provides access to AWS Neuron documentation through a local ChromaDB vector database for fast semantic search.

## Features

- **Local Vector Database**: Uses ChromaDB to store and search documentation locally
- **Semantic Search**: Powered by sentence-transformers for intelligent content matching  
- **Automatic Indexing**: Crawls and indexes AWS Neuron documentation automatically
- **Fast Retrieval**: No need to hit external websites for searches once indexed
- **Built with FastMCP**: Simplified, modern MCP server implementation

## Tools

### `index_neuron_docs`
Index AWS Neuron documentation into local ChromaDB for fast searching.

**Returns:** Status message about the indexing process

### `reindex_neuron_docs`  
Re-index AWS Neuron documentation, replacing existing data.

**Returns:** Status message about the re-indexing process

### `search_neuron_docs`
Search AWS Neuron documentation using semantic search.

**Parameters:**
- `query` (string, required): Search query for AWS Neuron documentation
- `max_results` (integer, optional): Maximum number of results to return (default: 10)

**Returns:** Formatted search results with titles, URLs, relevance scores, and snippets

### `get_neuron_doc_content`
Retrieve the full content of a specific AWS Neuron documentation page from local database or web.

**Parameters:**
- `url` (string, required): URL of the AWS Neuron documentation page to retrieve

### `list_neuron_guides`
List available AWS Neuron guides and tutorials by category from the local database.

**Parameters:**
- `category` (string, optional): Category to filter guides (default: "all")
  - Options: `all`, `neuron-guide`, `frameworks`, `neuron-runtime`, `neuron-compiler`, `tools`, `release-notes`

### `get_db_stats`
Get statistics about the local documentation database.

**Returns:** Database statistics including document count and sections

## Installation

### Option 1: Automated Setup (Recommended)

**Linux/macOS:**
```bash
./setup_venv.sh
```

**Windows:**
```cmd
setup_venv.bat
```

This will:
- Create a virtual environment in `venv/`
- Install all dependencies
- Install the package in development mode

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### With MCP Client

Add to your MCP configuration (make sure to use the virtual environment's Python):

```json
{
  "mcpServers": {
    "aws-neuron-docs": {
      "command": "/path/to/your/project/venv/bin/python",
      "args": [
        "-m",
        "aws_neuron_documentation_mcp_server.server"
      ],
      "env": {},
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### Direct Usage

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows

# Run the server
python -m aws_neuron_documentation_mcp_server.server
```

## Quick Start

1. **Set up the environment:**
   ```bash
   ./setup_venv.sh  # Linux/macOS
   # or
   setup_venv.bat   # Windows
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate.bat  # Windows
   ```

3. **Index the documentation (first time only):**
   ```bash
   python test_server.py
   ```
   This will download and index ~205 AWS Neuron documentation pages into a local ChromaDB.

4. **Use with your MCP client** - the database will be ready for fast semantic searches!

## Examples

### Index documentation (first time setup)
```json
{
  "tool": "index_neuron_docs",
  "arguments": {}
}
```

### Search for PyTorch tutorials with semantic matching
```json
{
  "tool": "search_neuron_docs", 
  "arguments": {
    "query": "PyTorch installation and setup",
    "max_results": 5
  }
}
```

### Get database statistics
```json
{
  "tool": "get_db_stats",
  "arguments": {}
}
```

### List framework-specific guides
```json
{
  "tool": "list_neuron_guides",
  "arguments": {
    "category": "frameworks"
  }
}
```

## Architecture

Built using **FastMCP**, a modern Python framework for creating MCP servers with:
- Simplified decorator-based tool definitions
- Automatic type validation and schema generation
- Reduced boilerplate code
- Better developer experience

## Requirements

- Python 3.8+
- fastmcp
- httpx
- beautifulsoup4
- lxml
- chromadb
- sentence-transformers
- aiofiles

## Virtual Environment Management

### Activating the Environment
```bash
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows

# Helper script (Linux/macOS only)
source activate_venv.sh
```

### Deactivating the Environment
```bash
deactivate
```

### Updating Dependencies
```bash
# Activate environment first
source venv/bin/activate

# Update packages
pip install --upgrade -r requirements.txt
```

## Database Location

The ChromaDB database is stored at `~/.aws_neuron_docs_mcp/chroma_db/` and persists between sessions.

## License

MIT License