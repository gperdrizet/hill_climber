#!/bin/bash

# Build and serve documentation
# This script builds the Sphinx documentation and starts an HTTP server

set -e

echo "Building documentation..."
cd /workspaces/hill_climber/docs
make clean
make html

echo "Starting documentation server on port 8000..."
cd /workspaces/hill_climber
nohup python -m http.server 8000 --directory docs/build/html > /tmp/docs-server.log 2>&1 & disown

echo "Documentation server started. Access it at http://localhost:8000"
