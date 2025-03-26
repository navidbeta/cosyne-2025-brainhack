import os
import sys
import io
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import json
import traceback
import fitz  # PyMuPDF for extracting images
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re

# Load environment variables
load_dotenv()

class PaperAnalyzer:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.client = None
        self.output_dir = 'paper_analysis_output'
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, 'figures'))
        
        # Get API key
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        print(f"API key found: {self.api_key[:5]}...")
        
        try:
            self.client = OpenAI(api_key=self.api_key)
            print("OpenAI client initialized successfully")
        except Exception as e:
            print(f"Error initializing OpenAI client: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            self.content = self._extract_pdf_content()
            print(f"PDF content extracted. Length: {len(self.content)} characters")
            
            # Extract figures
            self.figures = self._extract_figures()
            print(f"Extracted {len(self.figures)} figures from PDF")
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
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

    def _is_useful_image(self, img_data, page_num, text_content):
        """Determine if an image is likely to be useful (not decorative or irrelevant)."""
        
        # 1. Size filtering - extremely small images are likely icons or decorations
        MIN_WIDTH = 100
        MIN_HEIGHT = 100
        if img_data.get("width", 0) < MIN_WIDTH or img_data.get("height", 0) < MIN_HEIGHT:
            return False
            
        # 2. Format filtering - some formats are more likely to be decorative
        if img_data.get("format", "").lower() in ["png", "jpg", "jpeg"]:
            # For common formats, apply additional checks
            
            # Calculate image size in pixels
            total_pixels = img_data.get("width", 0) * img_data.get("height", 0)
            if total_pixels < 10000:  # Images with fewer than 10,000 pixels are likely decorative
                return False
        
        # 3. Check if this image's approximate position is referenced in the text
        # Look for references to figures around this page
        figure_pattern = r"fig(?:ure)?\.?\s*(\d+)"
        matches = re.finditer(figure_pattern, text_content, re.IGNORECASE)
        
        # Count references to figures near this page
        page_range = range(max(1, page_num-2), min(page_num+3, 1000))
        nearby_figures = []
        
        for match in matches:
            try:
                fig_num = int(match.group(1))
                if fig_num not in nearby_figures:
                    nearby_figures.append(fig_num)
            except:
                pass
        
        # If we found figure references near this page, this image is more likely to be useful
        if nearby_figures and len(nearby_figures) <= 5:  # If there are too many figure references, this heuristic is less reliable
            return True
            
        # 4. Check aspect ratio - extremely narrow or wide images are often decorative
        width = img_data.get("width", 0)
        height = img_data.get("height", 0)
        if width > 0 and height > 0:
            aspect_ratio = width / height
            if aspect_ratio > 5 or aspect_ratio < 0.2:  # Very wide or very tall
                return False
                
        # Default: Accept medium to large images
        if img_data.get("width", 0) > 200 and img_data.get("height", 0) > 200:
            return True
            
        return False  # Reject by default if no positive criteria met

    def _extract_figures(self):
        """Extract figures from the PDF using PyMuPDF (fitz)."""
        figures = []
        
        try:
            doc = fitz.open(self.pdf_path)
            
            # Extract figure captions first to help with filtering
            figure_captions = self._extract_figure_captions(self.content)
            
            for page_index, page in enumerate(doc):
                page_num = page_index + 1
                # Get page text for context
                page_text = page.get_text()
                
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    # Get image dimensions if available
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    image_ext = base_image.get("ext", "")
                    
                    # Create image data object for filtering
                    img_data = {
                        "page": page_num,
                        "index": img_index + 1,
                        "width": width,
                        "height": height,
                        "format": image_ext
                    }
                    
                    # Check if this image is likely useful
                    if self._is_useful_image(img_data, page_num, page_text):
                        image_bytes = base_image["image"]
                        
                        # Save the image
                        image_filename = f"page{page_num}_img{img_index+1}.{image_ext}"
                        image_path = os.path.join(self.output_dir, 'figures', image_filename)
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        img_data["path"] = image_path
                        figures.append(img_data)
                    else:
                        print(f"Skipping likely irrelevant image on page {page_num}")
            
            print(f"Filtered down to {len(figures)} potentially useful figures")
            
            # Map captions to figures based on page proximity and order
            for caption in figure_captions:
                # Try to find the closest figure to this caption
                best_match = None
                best_score = float('inf')
                
                figure_num = int(caption["figure_number"]) if caption["figure_number"].isdigit() else -1
                
                for i, figure in enumerate(figures):
                    if "figure_number" not in figure:  # Skip if already assigned
                        # Score based on figure number match and page proximity
                        # Lower score is better
                        score = abs(figure.get("page", 0) - (figure_num * 2))  # Page proximity weight
                        
                        if score < best_score:
                            best_score = score
                            best_match = i
                
                if best_match is not None:
                    figures[best_match]["caption"] = caption["caption"]
                    figures[best_match]["figure_number"] = caption["figure_number"]
            
            return figures
            
        except Exception as e:
            print(f"Error extracting figures: {str(e)}")
            traceback.print_exc()
            return []

    def _extract_figure_captions(self, content):
        """Extract figure captions from the text content."""
        # Improved regex to capture more variations of figure captions
        caption_pattern = r"(?:Fig(?:ure)?\.?\s*(\d+[a-zA-Z]?))[\.:]?\s*([^\n\r\.]{3,}(?:\.[^\n\r\.]{3,}){0,5})"
        matches = re.finditer(caption_pattern, content, re.IGNORECASE)
        
        captions = []
        for match in matches:
            figure_number = match.group(1)
            caption = match.group(2).strip()
            captions.append({
                "figure_number": figure_number,
                "caption": caption
            })
        
        print(f"Extracted {len(captions)} figure captions")
        return captions

    def _chunk_content(self, max_tokens=8000):
        """Split content into manageable chunks."""
        words = self.content.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1  # +1 for space
            if current_length > max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        print(f"Content split into {len(chunks)} chunks")
        
        # Return the first 10 chunks as requested (instead of 2)
        limited_chunks = chunks[:10]
        print(f"Limiting analysis to first {len(limited_chunks)} chunks")
        return limited_chunks

    def analyze_figures_and_tables(self):
        """Analyze the paper to extract figures, tables, and their methodologies with focus on data processing."""
        chunks = self._chunk_content()
        results = []

        for i, chunk in enumerate(chunks):
            print(f"Analyzing chunk {i+1}/{len(chunks)}")
            try:
                # Modified prompt to focus on data processing
                prompt = f"""Analyze the following text from an academic paper and identify:
1. Any figures or tables mentioned that involve data processing or analysis
2. The detailed methodology or data collection and processing methods associated with each figure/table (focus on computational steps, algorithms, software tools, data manipulation techniques, and statistical analyses)
3. The key findings or results presented in each figure/table that could be used for validation

ONLY extract information related to data processing, computational analysis, or algorithmic approaches.
Skip any figures/tables that don't involve significant data processing steps.

Format the response as JSON with the following structure:
{{
    "items": [
        {{
            "type": "figure/table",
            "identifier": "Figure/Table number",
            "data_processing_focus": true,
            "methodology": {{
                "description": "Detailed description of data processing methodology",
                "steps": ["Data processing step 1", "Data processing step 2", ...],
                "parameters": {{"param1": "value1", "param2": "value2", ...}},
                "algorithms": ["Algorithm 1", "Algorithm 2", ...],
                "software": ["Software 1", "Software 2", ...],
                "statistics": ["Statistical method 1", "Statistical method 2", ...]
            }},
            "findings": {{
                "key_results": ["Result 1", "Result 2", ...],
                "values": {{"metric1": "value1", "metric2": "value2", ...}},
                "interpretation": "Interpretation of results"
            }}
        }}
    ]
}}

Text chunk:
{chunk[:3000]}..."""

                # Test if the client is properly initialized
                if not self.client:
                    print("Error: OpenAI client is not initialized")
                    return []

                print("Sending request to OpenAI API...")
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a scientific paper analyzer specializing in extracting data processing methodologies, computational techniques, and algorithmic approaches from research papers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2
                    )
                    print("Response received from OpenAI API")
                    
                    content = response.choices[0].message.content
                    print(f"Response content: {content[:100]}...")
                    
                    try:
                        chunk_results = json.loads(content)
                        items = chunk_results.get('items', [])
                        print(f"Found {len(items)} items in response")
                        
                        # Filter to only include data processing related items
                        data_processing_items = [item for item in items if item.get('data_processing_focus', False) or 
                                                self._is_data_processing_related(item)]
                        print(f"Filtered to {len(data_processing_items)} data processing related items")
                        results.extend(data_processing_items)
                    except json.JSONDecodeError:
                        # Try to extract JSON part from text
                        json_match = re.search(r'({[\s\S]*})', content)
                        if json_match:
                            try:
                                extracted_json = json_match.group(1)
                                chunk_results = json.loads(extracted_json)
                                items = chunk_results.get('items', [])
                                data_processing_items = [item for item in items if item.get('data_processing_focus', False) or 
                                                        self._is_data_processing_related(item)]
                                results.extend(data_processing_items)
                            except:
                                print("Could not parse JSON even after extraction attempt")
                    
                except Exception as e:
                    print(f"Error in OpenAI request: {str(e)}")
                    traceback.print_exc()
                    continue
                    
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                traceback.print_exc()
                continue

        # Deduplicate results based on identifier
        unique_results = {}
        for item in results:
            identifier = item.get('identifier', '')
            if identifier and identifier not in unique_results:
                unique_results[identifier] = item
        
        return list(unique_results.values())
        
    def _is_data_processing_related(self, item):
        """Check if an item is related to data processing based on its content."""
        # Keywords related to data processing
        data_processing_keywords = [
            'algorithm', 'computation', 'processing', 'analysis', 'statistical', 
            'model', 'neural network', 'machine learning', 'training', 'dataset',
            'filter', 'parameter', 'optimization', 'cluster', 'classification',
            'regression', 'transform', 'feature', 'pipeline', 'preprocessing',
            'normalization', 'validation', 'evaluation', 'metrics', 'accuracy'
        ]
        
        # Check methodology description
        methodology = item.get('methodology', {})
        if isinstance(methodology, dict):
            description = methodology.get('description', '').lower()
            steps = ' '.join([str(s).lower() for s in methodology.get('steps', [])])
            software = ' '.join([str(s).lower() for s in methodology.get('software', [])])
            combined_text = description + ' ' + steps + ' ' + software
        else:
            combined_text = str(methodology).lower()
            
        # Check if any keywords are present
        for keyword in data_processing_keywords:
            if keyword.lower() in combined_text:
                return True
                
        return False

    def generate_replication_instructions(self, results):
        """Generate step-by-step instructions for replicating each figure based on the methodology."""
        replication_instructions = []
        
        for item in results:
            identifier = item.get('identifier', '')
            methodology = item.get('methodology', {})
            
            if isinstance(methodology, str):
                # Convert string methodology to structured format
                methodology = {
                    "description": methodology,
                    "steps": [],
                    "parameters": {},
                    "algorithms": [],
                    "software": [],
                    "statistics": []
                }
            
            prompt = f"""
Based on the following data processing methodology from a scientific paper, create detailed step-by-step programming instructions 
that would replicate the data processing pipeline for {item.get('type', 'figure/table')} {identifier}.

Methodology:
{json.dumps(methodology, indent=2)}

The instructions should:
1. Be specific and detailed enough for a data scientist to implement
2. Include all necessary data preparation, processing, and analysis steps
3. Specify what libraries or tools would be needed
4. Detail the computational algorithms and data transformations
5. Include any statistical or mathematical operations required
6. Describe how to generate the final output

Format your response as a numbered list of programming steps with code snippets where appropriate.
"""
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a scientific coding expert who helps researchers replicate data processing pipelines from scientific papers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                
                instructions = response.choices[0].message.content
                
                replication_instructions.append({
                    "identifier": identifier,
                    "type": item.get('type', 'figure/table'),
                    "instructions": instructions,
                    "original_methodology": methodology,
                    "findings_for_validation": item.get('findings', {})
                })
                
            except Exception as e:
                print(f"Error generating replication instructions for {identifier}: {str(e)}")
                traceback.print_exc()
        
        return replication_instructions

def main():
    try:
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not found in environment variables")
            print("Please create a .env file with your OpenAI API key")
            return

        print("Initializing PaperAnalyzer...")
        analyzer = PaperAnalyzer('pdf.pdf')
        print("Starting analysis...")
        results = analyzer.analyze_figures_and_tables()
        
        if not results:
            print("No results obtained from analysis")
            return
            
        print(f"Analysis complete. Found {len(results)} items.")
        
        # Generate replication instructions
        print("Generating replication instructions...")
        replication_instructions = analyzer.generate_replication_instructions(results)
        
        # Save all results to a JSON file
        output = {
            "paper_analysis": results,
            "figures_extracted": analyzer.figures,
            "replication_instructions": replication_instructions
        }
        
        with open(os.path.join(analyzer.output_dir, 'analysis_results.json'), 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {os.path.join(analyzer.output_dir, 'analysis_results.json')}")
        
        # Create a summary HTML file for easy viewing
        with open(os.path.join(analyzer.output_dir, 'summary.html'), 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Paper Analysis Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .item {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        .methodology {{ background-color: #f9f9f9; padding: 10px; }}
        .findings {{ background-color: #f0f8ff; padding: 10px; }}
        .replication {{ background-color: #f5f5f5; padding: 10px; white-space: pre-wrap; }}
        img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Paper Analysis: {os.path.basename(analyzer.pdf_path)}</h1>
    
    <h2>Extracted Figures ({len(analyzer.figures)})</h2>
    <div class="figures">""")
            
            # Generate figure HTML separately to avoid nested f-string issues
            figure_html = ""
            for i, fig in enumerate(analyzer.figures):
                if "path" in fig:
                    page_info = f"Unnamed (Page {fig.get('page', 0)})"
                    figure_num = fig.get("figure_number", page_info)
                    caption = fig.get("caption", "No caption available")
                    width = fig.get("width", 0)
                    height = fig.get("height", 0)
                    path = fig["path"]
                    figure_html += f'<div class="item"><h3>Figure {figure_num}</h3><img src="{path}" /><p>{caption}</p><p>Size: {width}x{height}</p></div>'
            
            f.write(figure_html)
            
            f.write("""
    </div>
    
    <h2>Analysis Results</h2>""")
            
            # Generate results HTML separately
            results_html = ""
            for item in results:
                item_type = item["type"]
                identifier = item["identifier"]
                methodology = json.dumps(item["methodology"], indent=2)
                findings = json.dumps(item["findings"], indent=2)
                
                results_html += f"""
    <div class="item">
        <h3>{item_type} {identifier}</h3>
        <div class="methodology">
            <h4>Methodology:</h4>
            <p>{methodology}</p>
        </div>
        <div class="findings">
            <h4>Findings:</h4>
            <p>{findings}</p>
        </div>
    </div>
    """
            
            f.write(results_html)
            
            f.write("""
    <h2>Replication Instructions</h2>""")
            
            # Generate replication instructions HTML separately
            replication_html = ""
            for item in replication_instructions:
                item_type = item["type"]
                identifier = item["identifier"]
                instructions = item["instructions"]
                
                replication_html += f"""
    <div class="item">
        <h3>{item_type} {identifier}</h3>
        <div class="replication">
            <h4>Instructions for Replication:</h4>
            <pre>{instructions}</pre>
        </div>
    </div>
    """
            
            f.write(replication_html)
            
            f.write("""
</body>
</html>""")
        
        print(f"Summary HTML created at {os.path.join(analyzer.output_dir, 'summary.html')}")
        
        # Print a summary
        print("\nSummary of findings:")
        for item in results:
            print(f"\n{item['type']} {item['identifier']}:")
            if isinstance(item['methodology'], dict):
                print(f"Methodology: {item['methodology'].get('description', '')[:100]}...")
            else:
                print(f"Methodology: {str(item['methodology'])[:100]}...")
            if isinstance(item['findings'], dict):
                print(f"Findings: {item['findings'].get('interpretation', '')[:100]}...")
            else:
                print(f"Findings: {str(item['findings'])[:100]}...")
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 