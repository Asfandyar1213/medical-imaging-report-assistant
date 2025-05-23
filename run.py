"""
Run the AI-Powered Medical Imaging Report Assistant
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

def check_dependencies():
    """
    Check if all dependencies are installed
    
    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    try:
        import streamlit
        import torch
        import monai
        import transformers
        import pydicom
        import cv2
        import numpy
        import pandas
        import matplotlib
        import sentence_transformers
        import chromadb
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def install_dependencies():
    """
    Install dependencies from requirements.txt
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def check_sample_images():
    """
    Check if sample images exist, download them if not
    """
    sample_images_dir = Path("data") / "sample_images"
    
    if not os.path.exists(sample_images_dir) or not os.listdir(sample_images_dir):
        print("Sample images not found. Downloading...")
        try:
            if os.path.exists("download_sample_images.py"):
                subprocess.check_call([sys.executable, "download_sample_images.py"])
            
            if os.path.exists("create_placeholder_image.py"):
                subprocess.check_call([sys.executable, "create_placeholder_image.py"])
        except subprocess.CalledProcessError as e:
            print(f"Error downloading sample images: {e}")

def run_streamlit():
    """
    Run the Streamlit application
    """
    print("Starting AI-Powered Medical Imaging Report Assistant...")
    
    # Open browser
    webbrowser.open("http://localhost:8501")
    
    # Run Streamlit
    subprocess.call(["streamlit", "run", "app.py"])

def main():
    """
    Main function to run the application
    """
    print("AI-Powered Medical Imaging Report Assistant")
    print("==========================================")
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("Installing dependencies...")
        if not install_dependencies():
            print("Failed to install dependencies. Please install them manually.")
            print("pip install -r requirements.txt")
            return
    
    # Check sample images
    check_sample_images()
    
    # Run Streamlit
    run_streamlit()

if __name__ == "__main__":
    main() 