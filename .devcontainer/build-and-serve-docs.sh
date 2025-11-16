#!/bin/bash
#
# build-and-serve-docs.sh
# 
# Purpose: Build Sphinx documentation and start a local HTTP server for viewing
# Usage: Called automatically by VS Code devcontainer's postAttachCommand
# When: Runs each time VS Code attaches to the container
#
# This script ensures fresh documentation is always available at http://localhost:8000
# and handles the documentation server lifecycle automatically.

# Exit immediately if any command fails (prevents partial builds)
set -e

echo "Building documentation..."

# Navigate to documentation directory
cd /workspaces/hill_climber/docs

# Remove old build artifacts to ensure clean rebuild
# This prevents stale content from previous builds
make clean

# Build HTML documentation using Sphinx
# Reads source .rst files and generates HTML in build/html/
# Warnings about duplicate objects or missing references will be shown
make html

echo "Starting documentation server on port 8000..."

# Return to project root for consistent working directory
cd /workspaces/hill_climber

# Start HTTP server to serve the built documentation:
# - nohup: Continue running even if terminal closes
# - python -m http.server 8000: Simple HTTP server on port 8000
# - --directory docs/build/html: Serve from the Sphinx output directory
# - > /tmp/docs-server.log 2>&1: Redirect stdout and stderr to log file
# - &: Run process in background
# - disown: Detach from shell to survive shell termination
nohup python -m http.server 8000 --directory docs/build/html > /tmp/docs-server.log 2>&1 & disown

echo "Documentation server started. Access it at http://localhost:8000"
echo "Server logs are available at /tmp/docs-server.log"
