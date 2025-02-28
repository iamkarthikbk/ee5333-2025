#!/bin/bash
# Script to update all submodules to the latest master

# Initialize submodules if not already done
git submodule init

# Update all submodules to their latest commit on master
git submodule update --remote --merge

echo "All submodules updated to latest master!"
