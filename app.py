"""
AI-Powered Medical Imaging Report Assistant
Main Streamlit application
"""

import os
import time
import streamlit as st
import numpy as np
import tempfile
from datetime import datetime
import uuid

# Import project modules
from models.image_processor import MedicalImageProcessor
from models.report_generator import ReportGenerator
from models.retrieval import RetrievalSystem
from utils.dicom_utils import read_dicom, read_image_file
from utils.visualization import highlight_anomalies, create_heatmap_overlay, plot_side_by_side
import config

# Set page configuration
st.set_page_config(
    page_title=config.STREAMLIT_TITLE,
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {
        "patient_id": "",
        "patient_name": "",
        "age": "",
        "gender": "",
        "history": "",
        "modality": "Chest X-ray"
    }

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "image_data" not in st.session_state:
    st.session_state.image_data = None
    
if "image_metadata" not in st.session_state:
    st.session_state.image_metadata = None
    
if "anomalies" not in st.session_state:
    st.session_state.anomalies = None
    
if "heatmap" not in st.session_state:
    st.session_state.heatmap = None
    
if "confidence_scores" not in st.session_state:
    st.session_state.confidence_scores = None
    
if "report" not in st.session_state:
    st.session_state.report = None
    
if "similar_cases" not in st.session_state:
    st.session_state.similar_cases = None

# Initialize models
@st.cache_resource
def load_models():
    image_processor = MedicalImageProcessor()
    report_generator = ReportGenerator()
    retrieval_system = RetrievalSystem()
    return image_processor, report_generator, retrieval_system

image_processor, report_generator, retrieval_system = load_models()

# App title
st.title("AI-Powered Medical Imaging Report Assistant")
st.markdown("Upload medical images to get AI-assisted analysis and report generation")

# Create sidebar for patient information
with st.sidebar:
    st.header("Patient Information")
    
    st.session_state.patient_info["patient_id"] = st.text_input(
        "Patient ID", 
        value=st.session_state.patient_info["patient_id"]
    )
    
    st.session_state.patient_info["patient_name"] = st.text_input(
        "Patient Name", 
        value=st.session_state.patient_info["patient_name"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.patient_info["age"] = st.text_input(
            "Age", 
            value=st.session_state.patient_info["age"]
        )
    with col2:
        st.session_state.patient_info["gender"] = st.selectbox(
            "Gender",
            options=["Male", "Female", "Other"],
            index=0 if st.session_state.patient_info["gender"] == "Male" else 
                  1 if st.session_state.patient_info["gender"] == "Female" else 
                  2 if st.session_state.patient_info["gender"] == "Other" else 0
        )
    
    st.session_state.patient_info["history"] = st.text_area(
        "Clinical History", 
        value=st.session_state.patient_info["history"]
    )
    
    st.session_state.patient_info["modality"] = st.selectbox(
        "Imaging Modality",
        options=["Chest X-ray", "CT Scan", "MRI", "Ultrasound"],
        index=0 if st.session_state.patient_info["modality"] == "Chest X-ray" else
              1 if st.session_state.patient_info["modality"] == "CT Scan" else
              2 if st.session_state.patient_info["modality"] == "MRI" else
              3 if st.session_state.patient_info["modality"] == "Ultrasound" else 0
    )

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Image Upload & Analysis", "Report Generation", "Similar Cases"])

# Tab 1: Image Upload & Analysis
with tab1:
    st.header("Upload Medical Image")
    
    uploaded_file = st.file_uploader(
        "Choose a DICOM, JPG, or PNG file",
        type=config.SUPPORTED_IMAGE_TYPES,
        help="Upload a medical image for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
        
        # Store uploaded file
        st.session_state.uploaded_image = tmp_filepath
        
        # Process image based on file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.dcm':
            # Read DICOM file
            image_data, metadata = read_dicom(tmp_filepath)
            
            # Update patient info from DICOM metadata if available
            if metadata:
                if not st.session_state.patient_info["patient_id"] and metadata.get("PatientID") != "Unknown":
                    st.session_state.patient_info["patient_id"] = metadata.get("PatientID")
                if not st.session_state.patient_info["patient_name"] and metadata.get("PatientName") != "Unknown":
                    st.session_state.patient_info["patient_name"] = metadata.get("PatientName")
                st.session_state.patient_info["modality"] = metadata.get("Modality", "Chest X-ray")
                
            st.session_state.image_metadata = metadata
        else:
            # Read regular image file
            image_data = read_image_file(tmp_filepath)
            st.session_state.image_metadata = None
        
        st.session_state.image_data = image_data
        
        # Display image
        if image_data is not None:
            st.image(image_data, caption="Uploaded Image", use_column_width=True)
            
            # Process image with AI
            with st.spinner("Analyzing image..."):
                # Detect anomalies
                anomalies, heatmap, confidence_scores = image_processor.detect_anomalies(image_data)
                
                # Store results in session state
                st.session_state.anomalies = anomalies
                st.session_state.heatmap = heatmap
                st.session_state.confidence_scores = confidence_scores
                
                # Create visualization
                if anomalies:
                    # Get bounding boxes, labels, and scores
                    bounding_boxes, labels, scores = image_processor.get_bounding_boxes_and_labels(anomalies)
                    
                    # Highlight anomalies
                    highlighted_image = highlight_anomalies(image_data, bounding_boxes, labels, scores)
                    
                    # Create heatmap overlay
                    heatmap_overlay = create_heatmap_overlay(image_data, heatmap)
                    
                    # Display side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(highlighted_image, caption="Detected Anomalies", use_column_width=True)
                    with col2:
                        st.image(heatmap_overlay, caption="Anomaly Heatmap", use_column_width=True)
                    
                    # Display findings
                    st.subheader("Findings")
                    for anomaly in anomalies:
                        st.metric(
                            label=anomaly["label"],
                            value=f"{anomaly['confidence']:.2f}",
                            delta="Detected" if anomaly["confidence"] > 0.7 else "Possible"
                        )
                else:
                    st.success("No anomalies detected")
        else:
            st.error("Error processing image")
    
    else:
        st.info("Please upload an image to begin")

# Tab 2: Report Generation
with tab2:
    st.header("Radiology Report")
    
    if st.session_state.image_data is not None and st.session_state.anomalies is not None:
        # Generate report if not already generated
        if st.session_state.report is None:
            with st.spinner("Generating report..."):
                # Prepare findings for report generator
                findings = {
                    "anomalies": st.session_state.anomalies,
                    "confidence_scores": st.session_state.confidence_scores
                }
                
                # Generate report
                st.session_state.report = report_generator.generate_full_report(
                    findings=findings,
                    patient_info=st.session_state.patient_info
                )
        
        # Display report
        if st.session_state.report:
            # Create expanders for each section
            for section, content in st.session_state.report.items():
                with st.expander(section, expanded=True):
                    st.write(content)
                    
                    # Add edit button for each section
                    edited_content = st.text_area(
                        f"Edit {section}",
                        value=content,
                        key=f"edit_{section}"
                    )
                    
                    # Update report if content changed
                    if edited_content != content:
                        st.session_state.report[section] = edited_content
            
            # Export options
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export as PDF"):
                    st.info("PDF export functionality would be implemented here")
            
            with col2:
                if st.button("Save to Patient Record"):
                    st.success("Report saved to patient record")
    else:
        st.info("Please upload and analyze an image first")

# Tab 3: Similar Cases
with tab3:
    st.header("Similar Cases")
    
    if st.session_state.image_data is not None and st.session_state.anomalies is not None:
        # Retrieve similar cases if not already retrieved
        if st.session_state.similar_cases is None:
            with st.spinner("Retrieving similar cases..."):
                # Create query from findings
                if st.session_state.report and "Findings" in st.session_state.report:
                    query_text = st.session_state.report["Findings"]
                elif st.session_state.anomalies:
                    query_text = ", ".join([a["label"] for a in st.session_state.anomalies])
                else:
                    query_text = "normal chest x-ray"
                
                # Retrieve similar cases
                st.session_state.similar_cases = retrieval_system.retrieve_similar_cases(
                    query_text=query_text,
                    n_results=3
                )
        
        # Display similar cases
        if st.session_state.similar_cases:
            for i, case in enumerate(st.session_state.similar_cases):
                with st.expander(f"Case {i+1}: {case['metadata']['diagnosis']}", expanded=True):
                    st.write(f"**Findings:** {case['findings']}")
                    st.write(f"**Patient:** {case['metadata']['patient_age']} year old {case['metadata']['patient_gender']}")
                    st.write(f"**Modality:** {case['metadata']['modality']}")
                    if case['similarity'] is not None:
                        st.progress(min(case['similarity'], 1.0))
                        st.write(f"Similarity: {case['similarity']:.2f}")
        else:
            st.info("No similar cases found")
    else:
        st.info("Please upload and analyze an image first")

# Footer
st.markdown("---")
st.caption("AI-Powered Medical Imaging Report Assistant - For demonstration purposes only")
st.caption("This tool is designed to assist medical professionals, not replace them.")

# Cleanup temporary files
def cleanup_temp_files():
    if st.session_state.uploaded_image and os.path.exists(st.session_state.uploaded_image):
        os.unlink(st.session_state.uploaded_image)

# Register cleanup function to run when the app exits
import atexit
atexit.register(cleanup_temp_files) 