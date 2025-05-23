"""
Download sample chest X-ray images for testing
"""

import os
import requests
from pathlib import Path

# Create directories if they don't exist
data_dir = Path("data")
sample_images_dir = data_dir / "sample_images"
os.makedirs(sample_images_dir, exist_ok=True)

# Sample chest X-ray images from open datasets
sample_images = [
    {
        "url": "https://openi.nlm.nih.gov/imgs/512/1/1_1.png",
        "filename": "normal_chest_xray.png",
        "description": "Normal chest X-ray"
    },
    {
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/covid-19-pneumonia-15-PA.jpg",
        "filename": "pneumonia_chest_xray.jpg",
        "description": "Pneumonia chest X-ray"
    },
    {
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/covid-19-pneumonia-30-PA.jpg",
        "filename": "covid_pneumonia_chest_xray.jpg",
        "description": "COVID-19 pneumonia chest X-ray"
    }
]

def download_image(url, filename):
    """
    Download an image from URL and save it to the specified filename
    
    Args:
        url: URL of the image
        filename: Path to save the image
    
    Returns:
        bool: Success or failure
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    """Download all sample images"""
    print("Downloading sample images...")
    
    for image in sample_images:
        filepath = sample_images_dir / image["filename"]
        
        if os.path.exists(filepath):
            print(f"File already exists: {filepath}")
            continue
        
        print(f"Downloading {image['description']} to {filepath}...")
        success = download_image(image["url"], filepath)
        
        if success:
            print(f"Successfully downloaded {filepath}")
        else:
            print(f"Failed to download {filepath}")
    
    print("Done!")

if __name__ == "__main__":
    main() 