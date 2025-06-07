Write-Host "Creating final commit and pushing to GitHub..." -ForegroundColor Green

# Add all files
git add .

# Create commit with detailed message
$commitMessage = @"
ISIT2025 Wireless Wanderlust: Final submission

- Added complete model implementations
- Added comprehensive documentation
- Added test suite and CI configuration
- Added submission package
- Added results and visualizations
- Updated README and requirements
- Cleaned up unnecessary files
"@

git commit -m $commitMessage

# Push to GitHub
git push

Write-Host "Done!" -ForegroundColor Green 