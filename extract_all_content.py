import os
import sys
import re
import json
import traceback
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ContentExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.output_dir = 'extracted_content'
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Extract PDF content
        try:
            self.content = self._extract_pdf_content()
            print(f"PDF content extracted. Length: {len(self.content)} characters")
        except Exception as e:
            print(f"Error extracting PDF content: {str(e)}")
            traceback.print_exc()
            sys.exit(1)

    def _extract_pdf_content(self):
        """Extract text content from PDF file."""
        try:
            reader = PdfReader(self.pdf_path)
            content = ""
            for i, page in enumerate(reader.pages):
                print(f"Extracting page {i+1}/{len(reader.pages)}")
                content += page.extract_text()
            return content
        except Exception as e:
            print(f"Error in _extract_pdf_content: {str(e)}")
            traceback.print_exc()
            raise

    def _extract_figure_captions(self):
        """Extract captions for all figures in the paper."""
        # Pattern to match figure captions like "Figure 1: This is a caption" or "Fig. 2. Caption text."
        caption_pattern = r"(?:Fig(?:ure)?\.?\s*(\d+[A-Za-z]?)[\.:])\s*([^\n\r\.]{3,}(?:\.[^\n\r\.]{3,}){0,5})"
        matches = re.finditer(caption_pattern, self.content, re.IGNORECASE)
        
        captions = {}
        for match in matches:
            figure_num = match.group(1)
            caption_text = match.group(2).strip()
            
            if figure_num not in captions:
                captions[figure_num] = caption_text
                
        return captions

    def _find_figures_in_text(self):
        """Find all figure references in the paper."""
        figure_pattern = r"Fig(?:ure)?\.?\s*(\d+[A-Za-z]?)"
        matches = re.finditer(figure_pattern, self.content, re.IGNORECASE)
        
        figures = {}
        for match in matches:
            fig_num = match.group(1)
            if fig_num not in figures:
                figures[fig_num] = []
                
            # Get context around the reference (200 chars before and after)
            start = max(0, match.start() - 200)
            end = min(len(self.content), match.end() + 200)
            context = self.content[start:end]
            
            figures[fig_num].append({
                "position": match.start(),
                "context": context
            })
        
        # Sort figures by number
        sorted_figures = {}
        for fig_num in sorted(figures.keys(), key=lambda x: (int(re.sub(r'[^0-9]', '', x)), x)):
            sorted_figures[fig_num] = figures[fig_num]
            
        return sorted_figures

    def extract_text_around_figure(self, figure_num, context_size=2000):
        """Extract text around all mentions of a specific figure."""
        figure_pattern = r"Fig(?:ure)?\.?\s*" + re.escape(figure_num)
        matches = re.finditer(figure_pattern, self.content, re.IGNORECASE)
        
        contexts = []
        for match in matches:
            start = max(0, match.start() - context_size)
            end = min(len(self.content), match.end() + context_size)
            context = self.content[start:end]
            contexts.append(context)
        
        return contexts

    def extract_all_content(self):
        """Extract all text content, figures, captions, and references."""
        # Extract figure captions
        captions = self._extract_figure_captions()
        print(f"Extracted {len(captions)} figure captions")
        
        # Find all figures
        all_figures = self._find_figures_in_text()
        print(f"Found {len(all_figures)} figures in the paper")
        
        # Prepare results structure
        results = {
            "full_text": self.content,
            "figures": {},
            "captions": captions
        }
        
        # Process each figure
        for figure_num, contexts in all_figures.items():
            # Extract extended text around the figure
            surrounding_text = self.extract_text_around_figure(figure_num, 2000)
            
            results["figures"][figure_num] = {
                "references": contexts,
                "extended_context": surrounding_text[:3], # Limit to first 3 extended contexts
                "caption": captions.get(figure_num, "")
            }
        
        # Save JSON results
        with open(os.path.join(self.output_dir, 'all_content.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"All content saved to {os.path.join(self.output_dir, 'all_content.json')}")
        return results

def main():
    try:
        # PDF path
        pdf_path = 'third.pdf'
        
        # Check if PDF exists
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return

        print("Initializing ContentExtractor...")
        extractor = ContentExtractor(pdf_path)
        
        # Extract all content
        print("Extracting all content...")
        extractor.extract_all_content()
        
        print("\nExtraction complete!")
        print(f"All content saved to {os.path.join(extractor.output_dir, 'all_content.json')}")
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 