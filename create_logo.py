"""
Create a simple logo for the Medical Imaging Report Assistant
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Create directories if they don't exist
docs_dir = Path("docs")
images_dir = docs_dir / "images"
os.makedirs(images_dir, exist_ok=True)

def create_logo(filename, size=(400, 400)):
    """
    Create a simple logo for the project
    
    Args:
        filename: Path to save the logo
        size: Size of the logo (width, height)
    """
    # Create a white background
    img = Image.new('RGB', size, color=(255, 255, 255))
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Draw a blue circle
    width, height = size
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3
    
    # Draw the outer circle (blue)
    draw.ellipse(
        [(center_x - radius, center_y - radius), 
         (center_x + radius, center_y + radius)], 
        fill=(41, 128, 185)
    )
    
    # Draw the inner circle (white)
    inner_radius = radius * 0.8
    draw.ellipse(
        [(center_x - inner_radius, center_y - inner_radius), 
         (center_x + inner_radius, center_y + inner_radius)], 
        fill=(255, 255, 255)
    )
    
    # Draw a simplified lung icon in the center
    lung_width = radius * 0.8
    lung_height = radius * 1.2
    
    # Left lung
    left_lung_x = center_x - lung_width * 0.5
    left_lung_y = center_y - lung_height * 0.4
    
    # Draw the left lung (light blue)
    draw.ellipse(
        [(left_lung_x - lung_width * 0.4, left_lung_y),
         (left_lung_x + lung_width * 0.4, left_lung_y + lung_height * 0.8)],
        fill=(133, 193, 233)
    )
    
    # Right lung
    right_lung_x = center_x + lung_width * 0.5
    right_lung_y = center_y - lung_height * 0.4
    
    # Draw the right lung (light blue)
    draw.ellipse(
        [(right_lung_x - lung_width * 0.4, right_lung_y),
         (right_lung_x + lung_width * 0.4, right_lung_y + lung_height * 0.8)],
        fill=(133, 193, 233)
    )
    
    # Draw a heart shape between the lungs (red)
    heart_x = center_x
    heart_y = center_y + lung_height * 0.1
    heart_size = lung_width * 0.3
    
    # Simple heart shape
    draw.ellipse(
        [(heart_x - heart_size, heart_y - heart_size),
         (heart_x, heart_y + heart_size * 0.5)],
        fill=(231, 76, 60)
    )
    
    draw.ellipse(
        [(heart_x, heart_y - heart_size),
         (heart_x + heart_size, heart_y + heart_size * 0.5)],
        fill=(231, 76, 60)
    )
    
    # Draw a triangle to complete the heart
    draw.polygon(
        [(heart_x - heart_size, heart_y),
         (heart_x + heart_size, heart_y),
         (heart_x, heart_y + heart_size * 1.2)],
        fill=(231, 76, 60)
    )
    
    # Add AI symbol (binary code) at the bottom
    binary_y = center_y + radius * 1.2
    binary_text = "01 AI 10"
    
    # Try to use a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", int(radius * 0.3))
    except IOError:
        font = ImageFont.load_default()
    
    # Get text size
    text_width = draw.textlength(binary_text, font=font)
    text_position = ((width - text_width) // 2, binary_y)
    
    # Draw text
    draw.text(text_position, binary_text, fill=(41, 128, 185), font=font)
    
    # Save the image
    img.save(filename)
    print(f"Created logo: {filename}")

def create_architecture_diagram(filename, size=(800, 500)):
    """
    Create a simple architecture diagram for the project
    
    Args:
        filename: Path to save the diagram
        size: Size of the diagram (width, height)
    """
    # Create a white background
    img = Image.new('RGB', size, color=(255, 255, 255))
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        box_font = ImageFont.truetype("arial.ttf", 18)
        arrow_font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        title_font = ImageFont.load_default()
        box_font = title_font
        arrow_font = title_font
    
    # Draw title
    title = "AI-Powered Medical Imaging Report Assistant - Architecture"
    title_width = draw.textlength(title, font=title_font)
    draw.text(((size[0] - title_width) // 2, 20), title, fill=(0, 0, 0), font=title_font)
    
    # Define colors
    blue = (41, 128, 185)
    green = (39, 174, 96)
    orange = (230, 126, 34)
    purple = (142, 68, 173)
    light_gray = (236, 240, 241)
    
    # Draw boxes
    box_height = 80
    box_width = 160
    margin = 40
    arrow_length = 60
    
    # Input box
    input_x = margin
    input_y = 100
    draw.rectangle([(input_x, input_y), (input_x + box_width, input_y + box_height)], 
                  fill=light_gray, outline=blue, width=2)
    input_text = "Medical Image Input"
    text_width = draw.textlength(input_text, font=box_font)
    draw.text((input_x + (box_width - text_width) // 2, input_y + box_height // 2 - 10), 
             input_text, fill=(0, 0, 0), font=box_font)
    
    # Image Processing box
    process_x = input_x + box_width + arrow_length
    process_y = input_y
    draw.rectangle([(process_x, process_y), (process_x + box_width, process_y + box_height)], 
                  fill=light_gray, outline=green, width=2)
    process_text = "Image Processing Module"
    text_width = draw.textlength(process_text, font=box_font)
    draw.text((process_x + (box_width - text_width) // 2, process_y + box_height // 2 - 10), 
             process_text, fill=(0, 0, 0), font=box_font)
    
    # Report Generation box
    report_x = process_x + box_width + arrow_length
    report_y = input_y
    draw.rectangle([(report_x, report_y), (report_x + box_width, report_y + box_height)], 
                  fill=light_gray, outline=orange, width=2)
    report_text = "Report Generation Module"
    text_width = draw.textlength(report_text, font=box_font)
    draw.text((report_x + (box_width - text_width) // 2, report_y + box_height // 2 - 10), 
             report_text, fill=(0, 0, 0), font=box_font)
    
    # Output box
    output_x = report_x + box_width + arrow_length
    output_y = input_y
    draw.rectangle([(output_x, output_y), (output_x + box_width, output_y + box_height)], 
                  fill=light_gray, outline=blue, width=2)
    output_text = "Final Report"
    text_width = draw.textlength(output_text, font=box_font)
    draw.text((output_x + (box_width - text_width) // 2, output_y + box_height // 2 - 10), 
             output_text, fill=(0, 0, 0), font=box_font)
    
    # Knowledge Base box
    kb_x = process_x + box_width // 2
    kb_y = process_y + box_height + arrow_length
    draw.rectangle([(kb_x, kb_y), (kb_x + box_width, kb_y + box_height)], 
                  fill=light_gray, outline=purple, width=2)
    kb_text = "Knowledge Base"
    text_width = draw.textlength(kb_text, font=box_font)
    draw.text((kb_x + (box_width - text_width) // 2, kb_y + box_height // 2 - 10), 
             kb_text, fill=(0, 0, 0), font=box_font)
    
    # Draw arrows
    # Input to Processing
    draw.line([(input_x + box_width, input_y + box_height // 2), 
               (process_x, input_y + box_height // 2)], 
              fill=(0, 0, 0), width=2)
    draw.polygon([(process_x, input_y + box_height // 2 - 5),
                 (process_x - 10, input_y + box_height // 2),
                 (process_x, input_y + box_height // 2 + 5)],
                fill=(0, 0, 0))
    
    # Processing to Report
    draw.line([(process_x + box_width, process_y + box_height // 2), 
               (report_x, process_y + box_height // 2)], 
              fill=(0, 0, 0), width=2)
    draw.polygon([(report_x, process_y + box_height // 2 - 5),
                 (report_x - 10, process_y + box_height // 2),
                 (report_x, process_y + box_height // 2 + 5)],
                fill=(0, 0, 0))
    
    # Report to Output
    draw.line([(report_x + box_width, report_y + box_height // 2), 
               (output_x, report_y + box_height // 2)], 
              fill=(0, 0, 0), width=2)
    draw.polygon([(output_x, report_y + box_height // 2 - 5),
                 (output_x - 10, report_y + box_height // 2),
                 (output_x, report_y + box_height // 2 + 5)],
                fill=(0, 0, 0))
    
    # Knowledge Base to Processing
    draw.line([(kb_x + box_width // 2, kb_y), 
               (kb_x + box_width // 2, process_y + box_height)], 
              fill=(0, 0, 0), width=2)
    draw.polygon([(kb_x + box_width // 2 - 5, process_y + box_height + 10),
                 (kb_x + box_width // 2, process_y + box_height),
                 (kb_x + box_width // 2 + 5, process_y + box_height + 10)],
                fill=(0, 0, 0))
    
    # Knowledge Base to Report
    draw.line([(kb_x + box_width, kb_y + box_height // 2), 
               (report_x + box_width // 2, report_y + box_height)], 
              fill=(0, 0, 0), width=2)
    draw.polygon([(report_x + box_width // 2 - 5, report_y + box_height + 10),
                 (report_x + box_width // 2, report_y + box_height),
                 (report_x + box_width // 2 + 5, report_y + box_height + 10)],
                fill=(0, 0, 0))
    
    # Add labels to arrows
    draw.text((input_x + box_width + 10, input_y + box_height // 2 - 20), 
             "DICOM/Images", fill=(0, 0, 0), font=arrow_font)
    
    draw.text((process_x + box_width + 10, process_y + box_height // 2 - 20), 
             "Anomalies", fill=(0, 0, 0), font=arrow_font)
    
    draw.text((report_x + box_width + 10, report_y + box_height // 2 - 20), 
             "Structured Report", fill=(0, 0, 0), font=arrow_font)
    
    draw.text((kb_x + box_width // 2 + 10, process_y + box_height + 10), 
             "Similar Cases", fill=(0, 0, 0), font=arrow_font)
    
    draw.text((report_x + box_width // 4, report_y + box_height + 10), 
             "Medical Knowledge", fill=(0, 0, 0), font=arrow_font)
    
    # Add technologies at the bottom
    tech_y = kb_y + box_height + 60
    tech_text = "Technologies: MONAI | PyTorch | Transformers | ChromaDB | Streamlit"
    text_width = draw.textlength(tech_text, font=box_font)
    draw.text(((size[0] - text_width) // 2, tech_y), tech_text, fill=(0, 0, 0), font=box_font)
    
    # Save the image
    img.save(filename)
    print(f"Created architecture diagram: {filename}")

def create_demo_gif(filename, size=(800, 500)):
    """
    Create a simple demo GIF placeholder
    
    Args:
        filename: Path to save the GIF
        size: Size of the image (width, height)
    """
    # Create frames for the GIF
    frames = []
    
    # Create 5 frames with different text
    texts = [
        "Upload Medical Image",
        "AI Analyzing Image...",
        "Anomalies Detected",
        "Generating Report...",
        "Report Ready for Review"
    ]
    
    for i, text in enumerate(texts):
        # Create a frame
        img = Image.new('RGB', size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Try to use a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 36)
            small_font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
            small_font = font
        
        # Draw progress bar
        progress_width = size[0] * 0.8
        progress_height = 30
        progress_x = (size[0] - progress_width) // 2
        progress_y = size[1] // 2 - progress_height // 2
        
        # Draw progress bar background
        draw.rectangle(
            [(progress_x, progress_y), (progress_x + progress_width, progress_y + progress_height)],
            outline=(41, 128, 185),
            width=2,
            fill=(236, 240, 241)
        )
        
        # Draw progress bar fill
        fill_width = progress_width * ((i + 1) / len(texts))
        draw.rectangle(
            [(progress_x, progress_y), (progress_x + fill_width, progress_y + progress_height)],
            fill=(41, 128, 185)
        )
        
        # Draw text
        text_width = draw.textlength(text, font=font)
        draw.text(
            ((size[0] - text_width) // 2, progress_y - 80),
            text,
            fill=(0, 0, 0),
            font=font
        )
        
        # Draw step number
        step_text = f"Step {i+1}/{len(texts)}"
        step_width = draw.textlength(step_text, font=small_font)
        draw.text(
            ((size[0] - step_width) // 2, progress_y + progress_height + 40),
            step_text,
            fill=(0, 0, 0),
            font=small_font
        )
        
        # Draw title
        title = "AI-Powered Medical Imaging Report Assistant"
        title_width = draw.textlength(title, font=small_font)
        draw.text(
            ((size[0] - title_width) // 2, 30),
            title,
            fill=(41, 128, 185),
            font=small_font
        )
        
        frames.append(img)
    
    # Save as GIF
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=1000,
        loop=0
    )
    print(f"Created demo GIF: {filename}")

def main():
    """Create images for the project"""
    # Create the logo
    logo_path = images_dir / "logo.png"
    create_logo(logo_path)
    
    # Create the architecture diagram
    architecture_path = images_dir / "architecture.png"
    create_architecture_diagram(architecture_path)
    
    # Create the demo GIF
    demo_path = images_dir / "demo.gif"
    create_demo_gif(demo_path)
    
    print("Done!")

if __name__ == "__main__":
    main() 