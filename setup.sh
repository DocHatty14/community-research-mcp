#!/bin/bash
set -e

echo "Setting up Community Research MCP..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install package in editable mode
echo "Installing dependencies..."
pip install -e .

# Copy example env if .env doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Copying .env.example to .env..."
        cp .env.example .env
        echo "✓ Setup complete!"
        echo ""
        echo "Next steps:"
        echo "1. Edit .env with your API keys"
        echo "2. Activate the virtual environment: source venv/bin/activate"
        echo "3. Run the server according to MCP documentation"
    else
        echo "Warning: .env.example not found. Create .env manually with your API keys."
    fi
else
    echo "✓ Setup complete! (.env already exists)"
fi
