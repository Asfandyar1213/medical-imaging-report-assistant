"""
Utility functions for handling DICOM medical image files
"""

import os
import pydicom
import numpy as np
from PIL import Image
import cv2

def read_dicom(file_path):
    """
    Read a DICOM file and return the image data and metadata
    
    Args:
        file_path: Path to the DICOM file
        
    Returns:
        tuple: (image_data, metadata_dict)
    """
    try:
        dicom = pydicom.dcmread(file_path)
        
        # Extract image data
        image_data = dicom.pixel_array
        
        # Apply windowing if available
        if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
            center = dicom.WindowCenter if not isinstance(dicom.WindowCenter, pydicom.multival.MultiValue) else dicom.WindowCenter[0]
            width = dicom.WindowWidth if not isinstance(dicom.WindowWidth, pydicom.multival.MultiValue) else dicom.WindowWidth[0]
            image_data = apply_windowing(image_data, center, width)
        
        # Normalize to 0-255 for display
        if image_data.max() > 0:
            image_data = (image_data / image_data.max() * 255).astype(np.uint8)
        
        # Extract metadata
        metadata = {
            'PatientID': getattr(dicom, 'PatientID', 'Unknown'),
            'PatientName': str(getattr(dicom, 'PatientName', 'Unknown')),
            'StudyDate': getattr(dicom, 'StudyDate', 'Unknown'),
            'Modality': getattr(dicom, 'Modality', 'Unknown'),
            'BodyPartExamined': getattr(dicom, 'BodyPartExamined', 'Unknown'),
            'StudyDescription': getattr(dicom, 'StudyDescription', 'Unknown'),
            'ImageComments': getattr(dicom, 'ImageComments', ''),
        }
        
        return image_data, metadata
    
    except Exception as e:
        print(f"Error reading DICOM file: {e}")
        return None, None

def apply_windowing(image, center, width):
    """
    Apply windowing to the image for better contrast
    
    Args:
        image: Image data
        center: Window center
        width: Window width
        
    Returns:
        numpy.ndarray: Windowed image
    """
    img_min = center - width // 2
    img_max = center + width // 2
    windowed_image = np.clip(image, img_min, img_max)
    windowed_image = ((windowed_image - img_min) / (img_max - img_min)) * 255.0
    return windowed_image

def save_as_png(image_data, output_path):
    """
    Save image data as PNG file
    
    Args:
        image_data: Image data as numpy array
        output_path: Path to save the PNG file
    """
    img = Image.fromarray(image_data)
    img.save(output_path)
    
def read_image_file(file_path):
    """
    Read an image file (JPG, PNG) and return the image data
    
    Args:
        file_path: Path to the image file
        
    Returns:
        numpy.ndarray: Image data
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in ['.dcm']:
        image_data, _ = read_dicom(file_path)
        return image_data
    else:
        # Read regular image file
        image_data = cv2.imread(file_path)
        if image_data is not None:
            # Convert from BGR to RGB
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        return image_data 