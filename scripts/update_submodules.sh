#!/bin/bash

# update_submodules.sh
# 
# This script initializes and updates all git submodules to their latest versions.
# It can be run from anywhere within the repository.
# 
# Usage: ./scripts/update_submodules.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print error messages
error() {
  echo -e "\033[0;31mERROR: $1\033[0m" >&2
  exit 1
}

# Function to print success messages
success() {
  echo -e "\033[0;32m$1\033[0m"
}

# Function to print info messages
info() {
  echo -e "\033[0;34m$1\033[0m"
}

# Navigate to the git repository root
cd "$(git rev-parse --show-toplevel)" || error "Failed to navigate to the git repository root"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  error "Not a git repository"
fi

info "Starting submodule update..."

# Initialize any submodules that haven't been initialized yet
info "Initializing submodules..."
git submodule init || error "Failed to initialize submodules"

# Update all submodules to their latest committed version
info "Updating submodules to the latest version..."
git submodule update --remote --merge || error "Failed to update submodules"

# Optionally, you can also run the following to pull changes with a rebase strategy
# uncomment if you prefer this approach
# git submodule update --remote --rebase || error "Failed to update submodules"

# Display status of submodules after update
info "Current submodule status:"
git submodule status

success "All submodules have been successfully updated to their latest versions!"
exit 0

