import os
import requests
from tqdm import tqdm
from typing import Optional

def download_file(url: str, save_path: str, chunk_size: int = 8192) -> None:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        save_path: Path to save the file
        chunk_size: Size of chunks to download
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)

def download_dichasus_dataset(save_dir: str = "data", force: bool = True) -> None:
    """
    Download the DICHASUS dataset files.
    
    Args:
        save_dir: Directory to save the files
        force: Whether to force download even if files exist
    """
    # Create data directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Base URL for the dataset
    base_url = "https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x/"
    
    # List of files to download
    files = [
        "dichasus-cf02.h5",
        "dichasus-cf03.h5",
        "dichasus-cf04.h5",
        "dichasus-cf05.h5",
        "dichasus-cf06.h5",
        "dichasus-cf07.h5"
    ]
    
    for file in files:
        url = f"{base_url}{file}"
        save_path = os.path.join(save_dir, file)
        if force or not os.path.exists(save_path):
            print(f"Downloading {file}...")
            download_file(url, save_path)
        else:
            print(f"{file} already exists, skipping...")

def download_kaggle_functions(save_dir: str = "src/utils", force: bool = True) -> None:
    """
    Download Kaggle utility functions.
    
    Args:
        save_dir: Directory to save the functions
        force: Whether to force download even if files exist
    """
    # Create utils directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # List of files to download with their raw GitHub URLs
    files = {
        "dichasus_evaluation.py": "https://raw.githubusercontent.com/tebs89/isit-2025-wireless-wanderlust/main/dichasus_evaluation.py",
        "dichasus_cf0x.py": "https://raw.githubusercontent.com/tebs89/isit-2025-wireless-wanderlust/main/dichasus_cf0x.py",
        "getting_started.py": "https://raw.githubusercontent.com/tebs89/isit-2025-wireless-wanderlust/main/getting_started.py"
    }
    
    for filename, url in files.items():
        save_path = os.path.join(save_dir, filename)
        if force or not os.path.exists(save_path):
            print(f"Downloading {filename}...")
            download_file(url, save_path)
        else:
            print(f"{filename} already exists, skipping...")

def main():
    """Main function to download all required data and functions."""
    print("Downloading DICHASUS dataset...")
    download_dichasus_dataset(force=True)
    
    print("\nDownloading Kaggle utility functions...")
    download_kaggle_functions(force=True)
    
    print("\nAll downloads completed!")

if __name__ == "__main__":
    main() 