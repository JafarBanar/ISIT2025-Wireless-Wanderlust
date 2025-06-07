#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting GitHub commit and push process..."

# Check if git is initialized
if [ ! -d .git ]; then
    echo -e "${RED}Git repository not initialized. Initializing...${NC}"
    git init
fi

# Add all files
echo "Adding files to git..."
git add .

# Create commit
echo "Creating commit..."
git commit -m "ISIT2025 Wireless Wanderlust: Final submission
- Basic model implementation (MAE: 0.3370)
- Trajectory-aware model
- Ensemble model
- Complete documentation
- Performance results
- Submission package"

# Check if remote exists
if ! git remote | grep -q "origin"; then
    echo -e "${RED}No remote repository found. Please add your GitHub repository URL:${NC}"
    read -p "Enter GitHub repository URL: " repo_url
    git remote add origin "$repo_url"
fi

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

# Check if push was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully pushed to GitHub!${NC}"
    echo "Repository URL: $(git remote get-url origin)"
else
    echo -e "${RED}Failed to push to GitHub. Please check your credentials and try again.${NC}"
    exit 1
fi

echo "Done!" 