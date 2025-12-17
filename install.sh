#!/bin/bash

echo "🔧 Installing GTA SA Map Converter environment..."

# Stop on error
set -e

# Set environment dir
VENV_DIR="venv"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creating virtual environment in '$VENV_DIR'..."
  python3 -m venv $VENV_DIR
else
  echo "📦 Virtual environment already exists."
fi

# Activate it
echo "🚀 Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
  echo "📄 Installing requirements from requirements.txt..."
  pip install -r requirements.txt
else
  echo "⚠️  requirements.txt not found!"
  exit 1
fi

echo "✅ Installation complete. You can now run:"
echo ""
echo "    source $VENV_DIR/bin/activate && python3 main.py"
echo ""
