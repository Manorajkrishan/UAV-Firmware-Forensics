#!/bin/bash

echo "========================================"
echo "Drone Firmware Detection Desktop App"
echo "========================================"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if Electron is installed
if [ ! -d "node_modules/electron" ]; then
    echo "Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "WARNING: Python not found. Backend may not start."
    echo "Please ensure Python is installed and in PATH."
    echo ""
fi

echo "Starting desktop application..."
echo ""
npm start


