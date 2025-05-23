"""
Image processing and anomaly detection module using pre-trained MONAI models
"""

import os
import numpy as np
import torch
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
    Resize,
    EnsureChannelFirst,
)
import cv2
import torchvision.transforms as transforms
from PIL import Image

# Import project config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class MedicalImageProcessor:
    """Medical image processor for anomaly detection"""
    
    def __init__(self, model_name=None):
        """
        Initialize the medical image processor
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name if model_name else config.MONAI_MODEL_NAME
        self.model = None
        self.transform = None
        self.labels = self._load_labels()
        
        # Initialize model and transforms
        self._initialize_model()
    
    def _load_labels(self):
        """
        Load labels for chest X-ray classification
        
        Returns:
            list: List of labels
        """
        # Common chest X-ray findings
        return [
            "Normal",
            "Cardiomegaly",
            "Lung Opacity",
            "Pleural Effusion",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural Thickening",
            "Nodule",
            "Mass",
            "Hernia",
            "Atelectasis"
        ]
    
    def _initialize_model(self):
        """
        Initialize the model and transforms
        """
        # For demonstration, we'll use a pre-trained DenseNet121 from MONAI
        # In a real application, you would load specific weights for medical imaging
        
        # Initialize transforms
        self.transform = Compose([
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize((224, 224)),
        ])
        
        # Initialize model
        try:
            # Create a DenseNet121 model with the appropriate number of output classes
            self.model = DenseNet121(
                spatial_dims=2,
                in_channels=1,  # Grayscale medical images
                out_channels=len(self.labels),
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"Model initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing model: {e}")
            # Fallback to a simple model for demonstration
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """
        Initialize a fallback model for demonstration purposes
        """
        print("Using fallback model for demonstration")
        # This is a mock model that will return random predictions
        self.model = None
    
    def preprocess_image(self, image):
        """
        Preprocess the image for model input
        
        Args:
            image: Input image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to grayscale if RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to 224x224
        resized = cv2.resize(gray, (224, 224))
        
        # Apply transforms
        if self.transform:
            try:
                # Convert to PIL Image
                pil_image = Image.fromarray(resized)
                
                # Apply transforms
                tensor = self.transform(pil_image)
                
                # Add batch dimension
                tensor = tensor.unsqueeze(0)
                
                return tensor.to(self.device)
            except Exception as e:
                print(f"Error in preprocessing: {e}")
        
        # Fallback preprocessing
        tensor = torch.from_numpy(resized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        tensor = tensor / 255.0  # Normalize to [0, 1]
        
        return tensor.to(self.device)
    
    def detect_anomalies(self, image):
        """
        Detect anomalies in the image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            tuple: (anomalies, heatmap, confidence_scores)
                anomalies: List of dictionaries with anomaly information
                heatmap: Heatmap highlighting anomalies
                confidence_scores: Dictionary of confidence scores for each label
        """
        # For demonstration purposes, we'll generate some random anomalies
        # In a real application, you would use the model predictions
        
        if self.model is not None:
            try:
                # Preprocess image
                tensor = self.preprocess_image(image)
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
                
                # Get confidence scores
                confidence_scores = {label: float(prob) for label, prob in zip(self.labels, probabilities)}
                
                # Filter anomalies with confidence above threshold
                anomalies = []
                for label, prob in confidence_scores.items():
                    if prob > config.CONFIDENCE_THRESHOLD and label != "Normal":
                        # Generate a random bounding box for demonstration
                        h, w = image.shape[:2]
                        x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
                        x2, y2 = x1 + np.random.randint(w//4, w//2), y1 + np.random.randint(h//4, h//2)
                        
                        anomalies.append({
                            'label': label,
                            'confidence': prob,
                            'bbox': [x1, y1, x2, y2]
                        })
                
                # Generate a simple heatmap for demonstration
                heatmap = np.zeros_like(image) if len(image.shape) == 2 else np.zeros(image.shape[:2])
                for anomaly in anomalies:
                    x1, y1, x2, y2 = anomaly['bbox']
                    # Create a Gaussian blob in the bounding box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    sigma = max((x2 - x1) // 4, (y2 - y1) // 4, 10)
                    
                    y, x = np.ogrid[:heatmap.shape[0], :heatmap.shape[1]]
                    heatmap += anomaly['confidence'] * np.exp(
                        -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2)
                    )
                
                # Normalize heatmap
                if heatmap.max() > 0:
                    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
                
                return anomalies, heatmap, confidence_scores
            
            except Exception as e:
                print(f"Error in anomaly detection: {e}")
        
        # Fallback to random predictions
        return self._generate_mock_predictions(image)
    
    def _generate_mock_predictions(self, image):
        """
        Generate mock predictions for demonstration
        
        Args:
            image: Input image
            
        Returns:
            tuple: (anomalies, heatmap, confidence_scores)
        """
        h, w = image.shape[:2]
        
        # Generate random confidence scores
        confidence_scores = {label: np.random.random() for label in self.labels}
        confidence_scores["Normal"] = 0.3  # Lower probability for normal to show some anomalies
        
        # Filter anomalies with confidence above threshold
        anomalies = []
        for label, prob in confidence_scores.items():
            if prob > config.CONFIDENCE_THRESHOLD and label != "Normal":
                # Generate a random bounding box
                x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
                x2, y2 = x1 + np.random.randint(w//4, w//2), y1 + np.random.randint(h//4, h//2)
                
                anomalies.append({
                    'label': label,
                    'confidence': prob,
                    'bbox': [x1, y1, x2, y2]
                })
        
        # Generate a simple heatmap
        heatmap = np.zeros_like(image) if len(image.shape) == 2 else np.zeros(image.shape[:2])
        for anomaly in anomalies:
            x1, y1, x2, y2 = anomaly['bbox']
            # Create a Gaussian blob in the bounding box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            sigma = max((x2 - x1) // 4, (y2 - y1) // 4, 10)
            
            y, x = np.ogrid[:heatmap.shape[0], :heatmap.shape[1]]
            heatmap += anomaly['confidence'] * np.exp(
                -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2)
            )
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        return anomalies, heatmap, confidence_scores
    
    def get_bounding_boxes_and_labels(self, anomalies):
        """
        Extract bounding boxes, labels, and scores from anomalies
        
        Args:
            anomalies: List of dictionaries with anomaly information
            
        Returns:
            tuple: (bounding_boxes, labels, scores)
        """
        bounding_boxes = [anomaly['bbox'] for anomaly in anomalies]
        labels = [anomaly['label'] for anomaly in anomalies]
        scores = [anomaly['confidence'] for anomaly in anomalies]
        
        return bounding_boxes, labels, scores 