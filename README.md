# AI-Powered Medical Imaging Report Assistant

This project provides an AI assistant that helps radiologists analyze medical images and generate preliminary reports. It combines computer vision models for anomaly detection with language models for report generation.

## Features

- Medical image processing and analysis (X-rays, CT scans, MRIs)
- Automatic anomaly detection and highlighting
- AI-generated preliminary radiology reports
- Similar case retrieval for reference
- User-friendly interface for radiologists

## Project Structure

- `app.py`: Main Streamlit application
- `models/`: Contains model loading and inference code
  - `image_processor.py`: Image preprocessing and anomaly detection
  - `report_generator.py`: Report generation using LLMs
  - `retrieval.py`: Similar case retrieval system
- `utils/`: Utility functions
  - `dicom_utils.py`: DICOM file handling
  - `visualization.py`: Image visualization tools
- `data/`: Sample data and knowledge base
- `config.py`: Configuration settings

## Setup and Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Upload a medical image (DICOM, JPG, PNG)
2. The system will process the image and detect potential anomalies
3. A preliminary report will be generated
4. Similar cases will be displayed for reference
5. Review, edit, and export the final report

## Models Used

- Image Analysis: MONAI's pre-trained models for medical imaging
- Report Generation: Hugging Face's medical-specific language models
- Retrieval System: Sentence transformers with ChromaDB

## Limitations

This tool is designed to assist medical professionals, not replace them. All generated reports should be reviewed by qualified radiologists before clinical use.

## License

MIT 