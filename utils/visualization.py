"""
Utility functions for visualizing medical images and anomalies
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image

def create_heatmap_overlay(image, heatmap, alpha=0.4):
    """
    Create a heatmap overlay on the original image
    
    Args:
        image: Original image (numpy array)
        heatmap: Heatmap of anomalies (numpy array)
        alpha: Transparency of the overlay (0-1)
        
    Returns:
        numpy.ndarray: Image with heatmap overlay
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Normalize heatmap to 0-255
    heatmap_normalized = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # Convert to RGB if needed
    if len(heatmap_colored.shape) > 2 and heatmap_colored.shape[2] == 3:
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image dimensions if needed
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlay

def highlight_anomalies(image, bounding_boxes, labels, scores, threshold=0.5):
    """
    Highlight anomalies on the image with bounding boxes and labels
    
    Args:
        image: Original image (numpy array)
        bounding_boxes: List of bounding boxes [x1, y1, x2, y2]
        labels: List of labels for each bounding box
        scores: List of confidence scores
        threshold: Minimum confidence score to display
        
    Returns:
        numpy.ndarray: Image with highlighted anomalies
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = image.copy()
    
    # Draw bounding boxes and labels
    for box, label, score in zip(bounding_boxes, labels, scores):
        if score >= threshold:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Prepare label text
            label_text = f"{label}: {score:.2f}"
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw background rectangle for text
            cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (255, 0, 0), -1)
            
            # Draw text
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def plot_side_by_side(original_image, processed_image, title1="Original", title2="Processed"):
    """
    Create a side-by-side plot of two images
    
    Args:
        original_image: Original image
        processed_image: Processed image
        title1: Title for the original image
        title2: Title for the processed image
        
    Returns:
        bytes: PNG image as bytes
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(processed_image, cmap='gray' if len(processed_image.shape) == 2 else None)
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    plt.close(fig)
    
    return buf.getvalue()

def create_image_grid(images, titles=None, cols=3):
    """
    Create a grid of images
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        cols: Number of columns in the grid
        
    Returns:
        bytes: PNG image as bytes
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < n_images:
            ax.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    plt.close(fig)
    
    return buf.getvalue() 