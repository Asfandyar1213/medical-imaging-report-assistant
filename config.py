"""
Configuration settings for the Medical Imaging Report Assistant
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "knowledge_base"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "sample_images"), exist_ok=True)

# Model configurations
MONAI_MODEL_NAME = "monai/resnet50-chest-xray"  # Pre-trained MONAI model for chest X-rays
MEDICAL_LLM_MODEL = "GanjinZero/biobart-v2-base"  # Medical domain-specific language model
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"  # Medical domain-specific embeddings

# Application settings
STREAMLIT_TITLE = "AI-Powered Medical Imaging Report Assistant"
STREAMLIT_DESCRIPTION = "Upload medical images to get AI-assisted analysis and report generation"
SUPPORTED_IMAGE_TYPES = ["dcm", "jpg", "jpeg", "png"]
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# Report generation settings
REPORT_SECTIONS = [
    "Clinical Information",
    "Technique",
    "Findings",
    "Impression",
    "Recommendations"
]

# Anomaly detection settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score to highlight anomalies

# Vector database settings
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db") 