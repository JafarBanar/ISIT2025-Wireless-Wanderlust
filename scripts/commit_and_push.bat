@echo off
echo Creating final commit and pushing to GitHub...

:: Add all files
git add .

:: Create commit with detailed message
git commit -m "ISIT2025 Wireless Wanderlust: Final submission

- Added complete model implementations
- Added comprehensive documentation
- Added test suite and CI configuration
- Added submission package
- Added results and visualizations
- Updated README and requirements"

:: Push to GitHub
git push

echo Done! 