# AWS Neuron Documentation MCP Server

An MCP (Model Context Protocol) server that provides access to AWS Neuron documentation through search and retrieval tools.

## Features

- **Search Documentation**: Search through AWS Neuron documentation for specific topics, APIs, or concepts
- **Retrieve Content**: Get the full content of specific documentation pages
- **List Guides**: Browse available guides and tutorials by category

## Tools

### `search_neuron_docs`
Search AWS Neuron documentation for specific topics.

**Parameters:**
- `query` (string, required): Search query for AWS Neuron documentation
- `max_results` (integer, optional): Maximum number of results to return (default: 10)

### `get_neuron_doc_content`
Retrieve the full content of a specific AWS Neuron documentation page.

**Parameters:**
- `url` (string, required): URL of the AWS Neuron documentation page to retrieve

### `list_neuron_guides`
List available AWS Neuron guides and tutorials by category.

**Parameters:**
- `category` (string, required): Category to filter guides
  - Options: `all`, `getting-started`, `tutorials`, `frameworks`, `inference`, `training`

## Installation

```bash
pip install aws-neuron-documentation-mcp-server
```

Or install from source:

```bash
git clone https://github.com/jgray-aws/neuron-mcp.git
cd aws-neuron-documentation-mcp-server
pip install -e .
```

## Usage

### With MCP Client

Add to your MCP configuration:

```json
{
  "mcpServers": {
        "aws-neuron-docs": {
      "command": "python",
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
aws-neuron-documentation-mcp-server
```

## Examples

### Search for PyTorch tutorials
```json
{
  "tool": "search_neuron_docs",
  "arguments": {
    "query": "PyTorch tutorial",
    "max_results": 5
  }
}
```

### Get content from a specific page
```json
{
  "tool": "get_neuron_doc_content",
  "arguments": {
    "url": "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/"
  }
}
```

### List getting started guides
```json
{
  "tool": "list_neuron_guides",
  "arguments": {
    "category": "getting-started"
  }
}
```

## Requirements

- Python 3.8+
- httpx
- beautifulsoup4
- lxml
- mcp
- pydantic

## License

MIT License