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
# This is safe to run multiple times
git lfs install

echo "Configuring Git LFS tracking patterns..."

# Track .db and .pkl files with Git LFS
# These commands update .gitattributes
git lfs track "*.db"
git lfs track "*.pkl"

echo "Pulling Git LFS files..."

# Pull all LFS files to replace pointer files with actual content
git lfs pull

echo "Git LFS setup complete!"
echo "Tracked patterns: *.db, *.pkl"
