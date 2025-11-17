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

# Determine the repository root dynamically
# This works regardless of where the repository is cloned
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building documentation..."

# Navigate to documentation directory
cd "$REPO_ROOT/docs"

# Create _static directory if it doesn't exist
mkdir -p source/_static

# Remove old build artifacts to ensure clean rebuild
# This prevents stale content from previous builds
make clean

# Build HTML documentation using Sphinx
# Reads source .rst files and generates HTML in build/html/
# Warnings about duplicate objects or missing references will be shown
make html

echo "Starting documentation server on port 8000..."

# Return to project root for consistent working directory
cd "$REPO_ROOT"

# Kill any existing process on port 8000
# Using lsof to target only processes bound to this specific port
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Start HTTP server to serve the built documentation:
# - setsid: Start in a new session to detach from parent
# - python -m http.server 8000: Simple HTTP server on port 8000
# - --directory docs/build/html: Serve from the Sphinx output directory
# - > /tmp/docs-server.log 2>&1: Redirect stdout and stderr to log file
# - &: Run process in background
# Using setsid ensures the process survives when the parent script exits
setsid python -m http.server 8000 --directory docs/build/html > /tmp/docs-server.log 2>&1 &

# Give the server a moment to start
sleep 1

echo "Documentation server started. Access it at http://localhost:8000"
echo "Server logs are available at /tmp/docs-server.log"
