# AI-Powered Medical Imaging Report Assistant

## Project Overview

This project provides an AI-powered assistant that helps radiologists analyze medical images and generate preliminary reports. It combines computer vision models for anomaly detection with language models for report generation, creating a comprehensive tool for medical imaging professionals.

## Key Features

1. **Medical Image Analysis**
   - Automatic detection of anomalies in chest X-rays and other medical images
   - Visualization of findings with bounding boxes and heatmaps
   - Confidence scores for detected conditions

2. **AI-Generated Reports**
   - Structured radiology reports with standard sections
   - Natural language descriptions of findings
   - Editable content for radiologist review

3. **Similar Case Retrieval**
   - Vector database of previous cases
   - Semantic search for similar findings
   - Reference cases to aid diagnosis

4. **User-Friendly Interface**
   - Clean, intuitive Streamlit web application
   - Patient information management
   - Export and save functionality

## Technical Components

### Models Used

1. **Image Analysis**
   - MONAI's DenseNet121 for chest X-ray classification
   - Fallback to demonstration mode if model loading fails

2. **Report Generation**
   - Hugging Face's biomedical language models
   - Section-specific text generation
   - Fallback templates for demonstration

3. **Retrieval System**
   - Sentence transformers for embedding generation
   - ChromaDB for vector storage and similarity search
   - Sample case database for demonstration

### Project Structure

```
.
├── app.py                  # Main Streamlit application
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
├── run.py                  # Application runner script
├── models/
│   ├── image_processor.py  # Image analysis module
│   ├── report_generator.py # Report generation module
│   └── retrieval.py        # Similar case retrieval module
├── utils/
│   ├── dicom_utils.py      # DICOM file handling utilities
│   └── visualization.py    # Image visualization utilities
└── data/
    ├── sample_images/      # Sample medical images
    └── knowledge_base/     # Vector database storage
```

## Deployment Instructions

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python run.py
   ```

3. Access the web interface at http://localhost:8501

## Limitations and Future Work

- The current implementation uses fallback mechanisms when models can't be loaded
- For a production environment, proper medical-specific models should be trained/fine-tuned
- Additional features could include:
  - Integration with hospital PACS systems
  - More comprehensive anomaly detection for various imaging modalities
  - User authentication and role-based access control
  - Long-term storage of reports and findings

## Ethical Considerations

This tool is designed to assist medical professionals, not replace them. All AI-generated content should be reviewed by qualified radiologists before clinical use. The system includes clear indications that outputs are AI-generated and require human verification. 