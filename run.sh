#!/bin/bash
# ANSI colors
NC='\033[0m'
WHITE='\033[1;37m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
GRAY='\033[0;90m'

cd "$(dirname "$0")"

# Check for venv
if [ ! -d "venv" ]; then
  echo -e "${CYAN}📦 Creating virtual environment...${NC}"
  python3 -m venv venv || { echo -e "${RED}❌ Failed to create venv${NC}"; exit 1; }
fi

# Activate venv
echo -e "${CYAN}⚙️  Activating virtual environment...${NC}"
source venv/bin/activate

# Run the app
echo -e "${GREEN}🚀 Running GTA SA Map Converter...${NC}"
echo ""

python3 main.py

# Exit handling
status=$?
if [ $status -eq 0 ]; then
  echo -e "${GREEN}✅ Application exited successfully.${NC}"
else
  echo -e "${RED}❌ Application exited with errors (code $status).${NC}"
fi

exit $status
