import os
import sys
import fitz  # PyMuPDF
import re
from pathlib import Path
import logging
from tqdm import tqdm
from PIL import Image
import io

class PaperFigureExtractor:
    def __init__(self, output_dir="extracted_figures"):
        self.output_dir = output_dir
        self.setup_logging()
        
        # Minimum dimensions for a figure (in pixels)
        self.min_width = 200
        self.min_height = 200
        
        # Maximum dimensions for a figure (to filter out full-page images)
        self.max_width = 20000
        self.max_height = 20000
        
        # Minimum file size (in bytes) to be considered a figure
        self.min_file_size = 10000  # 10KB
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def is_valid_figure(self, image_bytes, width, height, ext):
        """Check if the image is a valid figure based on size and content."""
        try:
            # Check dimensions
            if width < self.min_width or height < self.min_height:
                return False
            if width > self.max_width or height > self.max_height:
                return False
            
            # Check file size
            if len(image_bytes) < self.min_file_size:
                return False
            
            # For PNG and JPEG images, analyze the image content
            if ext.lower() in ['png', 'jpg', 'jpeg']:
                # Open image with PIL
                img = Image.open(io.BytesIO(image_bytes))
                
                # Convert to RGB if necessary
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                
                # Get image statistics
                img_array = img.convert('RGB')
                r, g, b = img_array.split()
                
                # Calculate color variance (to detect icons/logos with flat colors)
                r_var = r.getextrema()[1] - r.getextrema()[0]
                g_var = g.getextrema()[1] - g.getextrema()[0]
                b_var = b.getextrema()[1] - b.getextrema()[0]
                
                # If all color channels have very low variance, it's likely an icon/logo
                if r_var < 50 and g_var < 50 and b_var < 50:
                    return False
                
                # Calculate aspect ratio
                aspect_ratio = width / height
                
                # Filter out very tall or wide images (likely decorative elements)
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error analyzing image: {str(e)}")
            return False
    
    def extract_figures(self, pdf_path):
        """Extract figures from the PDF file."""
        try:
            self.logger.info("Opening PDF file")
            doc = fitz.open(pdf_path)
            
            # Create figures directory
            figures_dir = os.path.join(self.output_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            # Extract figures from each page
            figures = []
            skipped = 0
            
            for page_num in tqdm(range(len(doc)), desc="Extracting figures"):
                page = doc[page_num]
                
                # Get all images on the page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Get image extension
                    ext = base_image["ext"]
                    
                    # Get image dimensions
                    width = img[2]
                    height = img[3]
                    
                    # Check if this is a valid figure
                    if not self.is_valid_figure(image_bytes, width, height, ext):
                        skipped += 1
                        continue
                    
                    # Generate figure filename
                    figure_filename = f"figure_page{page_num+1}_{img_index+1}.{ext}"
                    figure_path = os.path.join(figures_dir, figure_filename)
                    
                    # Save the image
                    with open(figure_path, "wb") as f:
                        f.write(image_bytes)
                    
                    figures.append({
                        "page": page_num + 1,
                        "index": img_index + 1,
                        "filename": figure_filename,
                        "path": figure_path,
                        "dimensions": f"{width}x{height}",
                        "format": ext,
                        "file_size": len(image_bytes)
                    })
            
            doc.close()
            self.logger.info(f"Skipped {skipped} non-figure images")
            return figures
            
        except Exception as e:
            self.logger.error(f"Error extracting figures: {str(e)}")
            raise
    
    def extract_figure_captions(self, pdf_path):
        """Extract figure captions from the PDF."""
        try:
            self.logger.info("Extracting figure captions")
            doc = fitz.open(pdf_path)
            
            captions = []
            caption_pattern = re.compile(r'Figure\s+\d+[.:]\s+.*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE)
            
            for page_num in tqdm(range(len(doc)), desc="Extracting captions"):
                page = doc[page_num]
                text = page.get_text()
                
                # Find all figure captions on the page
                matches = caption_pattern.finditer(text)
                for match in matches:
                    caption = match.group().strip()
                    captions.append({
                        "page": page_num + 1,
                        "caption": caption
                    })
            
            doc.close()
            return captions
            
        except Exception as e:
            self.logger.error(f"Error extracting captions: {str(e)}")
            raise
    
    def process_paper(self, pdf_path):
        """Process a paper: extract figures and captions."""
        try:
            # Extract figures
            figures = self.extract_figures(pdf_path)
            
            # Extract captions
            captions = self.extract_figure_captions(pdf_path)
            
            # Create summary
            summary = {
                "pdf_path": pdf_path,
                "figures": figures,
                "captions": captions,
                "total_figures": len(figures),
                "total_captions": len(captions)
            }
            
            # Save summary to JSON
            import json
            summary_path = os.path.join(self.output_dir, "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Successfully processed paper. Found {len(figures)} figures and {len(captions)} captions.")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error processing paper: {str(e)}")
            raise

def main():
    pdf_path = "third.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)
    
    extractor = PaperFigureExtractor()
    
    try:
        summary = extractor.process_paper(pdf_path)
        print("\nProcessing complete!")
        print(f"Figures saved in: {os.path.join(extractor.output_dir, 'figures')}")
        print(f"Summary saved in: {os.path.join(extractor.output_dir, 'summary.json')}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 