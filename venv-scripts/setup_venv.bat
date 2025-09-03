@echo off
REM Setup script for AWS Neuron Documentation MCP Server (Windows)
REM This script creates a virtual environment and installs dependencies

echo 🚀 Setting up AWS Neuron Documentation MCP Server
echo ==================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed.
    pause
    exit /b 1
)

echo ✅ Found Python
python --version

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Install the package in development mode
echo 🔨 Installing package in development mode...
pip install -e .

echo.
echo ✅ Setup complete!
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To test the server, run:
echo   python test_server.py
echo.
echo To deactivate the virtual environment, run:
echo   deactivate
echo.
pause