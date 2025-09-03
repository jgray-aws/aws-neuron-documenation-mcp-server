#!/bin/bash

# Clean up script for AWS Neuron Documentation MCP Server
# This script removes the virtual environment

echo "🧹 Cleaning up virtual environment..."

if [ -d "venv" ]; then
    echo "📦 Removing virtual environment directory..."
    rm -rf venv
    echo "✅ Virtual environment removed!"
else
    echo "ℹ️  No virtual environment found to clean up."
fi

echo ""
echo "To recreate the virtual environment, run:"
echo "  ./setup_venv.sh"