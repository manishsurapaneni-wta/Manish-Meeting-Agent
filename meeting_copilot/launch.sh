#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Meeting Copilot Setup...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 is not installed. Please install pip3 and try again.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Install/upgrade pip
echo -e "${YELLOW}📦 Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p audio output

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating template...${NC}"
    echo "# OpenAI API Key" > .env
    echo "OPENAI_API_KEY=your_api_key_here" >> .env
    echo "# Hugging Face Token (for speaker diarization)" >> .env
    echo "HF_TOKEN=your_hf_token_here" >> .env
    echo -e "${RED}⚠️  Please edit .env file with your API keys before continuing${NC}"
    exit 1
fi

# Check if API keys are set
if grep -q "your_api_key_here" .env || grep -q "your_hf_token_here" .env; then
    echo -e "${RED}⚠️  Please set your API keys in the .env file before continuing${NC}"
    exit 1
fi

# Function to open browser
open_browser() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open http://localhost:8000
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open http://localhost:8000
    elif [[ "$OSTYPE" == "msys" ]]; then
        # Windows
        start http://localhost:8000
    fi
}

# Start the server
echo -e "${GREEN}🚀 Starting Meeting Copilot server...${NC}"
echo -e "${YELLOW}📝 Server logs will appear below. Press Ctrl+C to stop the server.${NC}"
echo -e "${GREEN}🌐 Opening browser...${NC}"

# Open browser after a short delay
(sleep 2 && open_browser) &

# Start the server
uvicorn web_app:app --reload --host 0.0.0.0 --port 8000 