#!/bin/bash

# Helper script to activate the virtual environment
# Usage: source activate_venv.sh

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup_venv.sh first."
    return 1
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"

# Check if the package is installed
if python -c "import aws_neuron_documentation_mcp_server" 2>/dev/null; then
    echo "✅ AWS Neuron Documentation MCP Server is installed"
else
    echo "⚠️  Package not found. You may need to run: pip install -e ."
fi