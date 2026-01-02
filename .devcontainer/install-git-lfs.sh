#!/bin/bash
#
# install-git-lfs.sh
# 
# Purpose: Install and configure Git LFS in the development container
# Usage: Called automatically by VS Code devcontainer's onCreateCommand
# When: Runs once during initial container creation
#
# This script:
# - Installs Git LFS
# - Initializes Git LFS in the repository
# - Configures tracking patterns for large binary files

# Exit immediately if any command fails
set -e

echo "Installing Git LFS..."

# Install Git LFS from apt repository
sudo apt-get install git-lfs -y

echo "Initializing Git LFS in repository..."

# Initialize Git LFS (sets up hooks and config)
# Use --force to overwrite any existing hooks if needed
git lfs install --force || git lfs install || echo "Warning: git lfs install had issues, continuing anyway"

# Check if we're in a git repository before running git commands
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Configuring Git LFS tracking patterns..."

    # Track .db and .pkl files with Git LFS
    # These commands update .gitattributes
    git lfs track "*.db"
    git lfs track "*.pkl"

    echo "Pulling Git LFS files..."

    # Pull all LFS files to replace pointer files with actual content
    git lfs pull || echo "Warning: git lfs pull failed (this is OK if there are no LFS files yet)"

    echo "Git LFS setup complete!"
    echo "Tracked patterns: *.db, *.pkl"
else
    echo "Warning: Not in a git repository. Skipping LFS tracking configuration."
    echo "Run 'git lfs track' manually after initializing the repository."
fi
