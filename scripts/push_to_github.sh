#!/bin/bash

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    git init
fi

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: ISIT2025 Wireless Wanderlust submission"

# Add GitHub remote (replace with your repository URL)
git remote add origin https://github.com/yourusername/ISIT2025.git

# Push to GitHub
git push -u origin main

echo "Project pushed to GitHub successfully!" 