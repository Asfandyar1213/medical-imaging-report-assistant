"""
Create a placeholder chest X-ray image for testing
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Create directories if they don't exist
data_dir = Path("data")
sample_images_dir = data_dir / "sample_images"
os.makedirs(sample_images_dir, exist_ok=True)

def create_placeholder_image(filename, text="Normal Chest X-ray", size=(512, 512)):
    """
    Create a placeholder image with text
    
    Args:
        filename: Path to save the image
        text: Text to display on the image
        size: Size of the image (width, height)
    """
    # Create a grayscale image
    img = Image.new('L', size, color=0)
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Draw a lung-like shape
    width, height = size
    center_x, center_y = width // 2, height // 2
    
    # Draw the ribcage outline
    for i in range(10):
        y_offset = -100 + i * 30
        ellipse_width = 300 - i * 10
        ellipse_height = 40
        left = center_x - ellipse_width // 2
        top = center_y + y_offset - ellipse_height // 2
        right = center_x + ellipse_width // 2
        bottom = center_y + y_offset + ellipse_height // 2
        
        draw.arc([left, top, right, bottom], 0, 360, fill=180, width=2)
    
    # Draw the spine
    draw.rectangle([center_x - 10, center_y - 150, center_x + 10, center_y + 150], fill=200)
    
    # Draw the left lung
    draw.ellipse([center_x - 180, center_y - 120, center_x - 40, center_y + 120], outline=220, width=2)
    
    # Draw the right lung
    draw.ellipse([center_x + 40, center_y - 120, center_x + 180, center_y + 120], outline=220, width=2)
    
    # Add some texture to make it look more like an X-ray
    pixels = np.array(img)
    noise = np.random.normal(0, 10, size).astype(np.uint8)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    # Add text
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    # Get text size
    text_width = draw.textlength(text, font=font)
    text_position = ((width - text_width) // 2, height - 50)
    
    # Draw text
    draw.text(text_position, text, fill=255, font=font)
    
    # Save the image
    img.save(filename)
    print(f"Created placeholder image: {filename}")

def main():
    """Create placeholder images"""
    normal_xray_path = sample_images_dir / "normal_chest_xray.png"
    
    # Create a placeholder for the normal chest X-ray
    create_placeholder_image(normal_xray_path)

if __name__ == "__main__":
    main() 