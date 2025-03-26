import os
import sys
import re
import json
import traceback
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DirectPaperInstructions:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.client = None
        self.output_dir = 'direct_reproduction_instructions'
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
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
        
        # Extract PDF content
        try:
            self.content = self._extract_pdf_content()
            print(f"PDF content extracted. Length: {len(self.content)} characters")
        except Exception as e:
            print(f"Error extracting PDF content: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
        
        # Find figures and methods sections
        self.sections = self._identify_paper_sections()
        print(f"Paper sections identified: {', '.join(self.sections.keys())}")

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

    def _identify_paper_sections(self):
        """Identify the main sections of the paper."""
        prompt = f"""
        Analyze the following academic paper and identify its main sections (especially Methods, Results, Materials and Methods, etc.).
        For each section, provide the section title and the approximate start and end position in the text.
        
        Format your response as a JSON object with this structure:
        {{
            "section_name": {{
                "start": start_position,
                "end": end_position
            }},
            ...
        }}
        
        Paper text (first 10000 chars):
        {self.content[:10000]}...
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a scientific paper analyzer that identifies the structure of academic papers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            try:
                # Extract JSON from the response
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    sections = json.loads(json_match.group(1))
                    return sections
                else:
                    print("Could not extract sections JSON from response")
                    return {}
            except Exception as e:
                print(f"Error parsing sections JSON: {str(e)}")
                return {}
                
        except Exception as e:
            print(f"Error identifying sections: {str(e)}")
            return {}

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

    def _extract_methods_for_figure(self, figure_num, figure_contexts):
        """Extract the methods used to produce a specific figure."""
        # Get Methods section if available
        methods_text = ""
        for section_name, section_range in self.sections.items():
            if "method" in section_name.lower() or "material" in section_name.lower():
                start = section_range.get("start", 0)
                end = section_range.get("end", len(self.content))
                methods_text += self.content[start:end] + "\n\n"
        
        # If methods section is too short, use the whole paper
        if len(methods_text) < 1000:
            methods_text = self.content
        
        # Find specific method for this figure
        prompt = f"""
        I need to extract the EXACT methodology used to produce Figure {figure_num} in this scientific paper.
        
        Here are some contexts where Figure {figure_num} is mentioned:
        {json.dumps([ctx["context"] for ctx in figure_contexts[:3]], indent=2)}
        
        Please examine the Methods section below and extract EXACTLY the text that describes:
        1. The data collection or generation process for Figure {figure_num}
        2. The data processing steps applied specifically for Figure {figure_num}
        3. The algorithms, statistical methods, or computational techniques used
        4. The software or tools used for creating Figure {figure_num}
        5. Any parameters or settings mentioned
        
        Only include text DIRECTLY FROM THE PAPER that relates to the methodology for Figure {figure_num}.
        If the methods section doesn't explicitly mention Figure {figure_num}, look for related experiments or analyses that would have produced the figure.
        
        Methods section:
        {methods_text[:10000]}...
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using GPT-4o-mini
                messages=[
                    {"role": "system", "content": "You are a scientific paper analyzer that extracts the exact methodology text from papers. Only return text found directly in the paper, don't make up or summarize methods."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            extracted_method = response.choices[0].message.content
            return extracted_method
            
        except Exception as e:
            print(f"Error extracting methods for Figure {figure_num}: {str(e)}")
            return "Error extracting methods."

    def _generate_reproduction_instructions(self, figure_num, methodology_text):
        """Generate step-by-step instructions to reproduce a figure based on the extracted methodology."""
        prompt = f"""
        Based on the FOLLOWING EXACT METHODOLOGY TEXT extracted from a scientific paper for Figure {figure_num}, 
        create detailed step-by-step instructions that would allow someone to reproduce the figure.
        
        METHODOLOGY FROM PAPER:
        {methodology_text}
        
        Please provide:
        1. A complete list of required software, libraries, packages, and versions needed (if mentioned in the text)
        2. Any datasets required, with information on how to obtain them (exactly as mentioned in the text)
        3. Detailed preprocessing steps, EXACTLY as described in the methodology
        4. All algorithm parameters, statistical methods, and computational procedures SPECIFICALLY mentioned
        5. Step-by-step code instructions (using pseudocode where appropriate)
        6. Visualization/figure generation instructions based on the information provided
        
        Format your response as a numbered list of steps. 
        BE EXTREMELY CAREFUL to only use information explicitly stated in the extracted methodology text.
        Do not invent or assume methods not mentioned in the text.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using GPT-4o-mini
                messages=[
                    {"role": "system", "content": "You are a scientific reproduction expert that creates detailed instructions for reproducing scientific figures based solely on the methodology text provided. Never invent steps or details not explicitly mentioned in the methodology text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            instructions = response.choices[0].message.content
            return instructions
            
        except Exception as e:
            print(f"Error generating reproduction instructions for Figure {figure_num}: {str(e)}")
            return "Error generating instructions."

    def generate_all_instructions(self, max_figures=5):
        """Generate reproduction instructions for all figures in the paper."""
        # Find all figures
        all_figures = self._find_figures_in_text()
        print(f"Found {len(all_figures)} figures in the paper")
        
        # Limit to max_figures
        figures_to_process = list(all_figures.items())[:max_figures]
        
        # Process each figure
        results = []
        for figure_num, contexts in figures_to_process:
            print(f"Processing Figure {figure_num}...")
            
            # Extract methodology text for this figure
            methodology_text = self._extract_methods_for_figure(figure_num, contexts)
            
            # Generate reproduction instructions
            instructions = self._generate_reproduction_instructions(figure_num, methodology_text)
            
            results.append({
                "figure_number": figure_num,
                "raw_methodology_text": methodology_text,
                "reproduction_instructions": instructions,
                "contexts": contexts[:3]  # Include only first 3 contexts for brevity
            })
            
        return results

    def generate_output(self, results):
        """Generate HTML and JSON output for the results."""
        # Save JSON results
        with open(os.path.join(self.output_dir, 'direct_reproduction_instructions.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create HTML summary
        with open(os.path.join(self.output_dir, 'direct_reproduction_instructions.html'), 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Direct Reproduction Instructions from Paper Text</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .figure { 
            border: 1px solid #ddd; 
            padding: 20px; 
            margin-bottom: 30px; 
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .method-text { 
            background-color: #f0f8ff; 
            padding: 15px; 
            margin: 15px 0;
            border-left: 4px solid #4285f4;
            white-space: pre-wrap;
            font-size: 0.9em;
        }
        .instructions { 
            background-color: #f5f5f5; 
            padding: 20px; 
            margin-top: 15px;
            border-left: 4px solid #ea4335;
            white-space: pre-wrap;
        }
        .context-block {
            background-color: #f9f9f9;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #0f9d58;
            font-size: 0.9em;
        }
        code {
            background-color: #f6f8fa;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: monospace;
        }
        pre {
            background-color: #f6f8fa;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: monospace;
        }
        h1 { color: #4285f4; }
        h2 { color: #4285f4; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        h3 { color: #4285f4; }
        h4 { color: #0f9d58; }
    </style>
</head>
<body>
    <h1>Direct Reproduction Instructions from Paper Text</h1>
    <p>This document contains methodology text extracted directly from the paper and step-by-step instructions for reproducing each figure.</p>
""")
            
            # Add each figure
            for result in results:
                figure_num = result["figure_number"]
                
                # Format context blocks
                contexts_html = ""
                for i, context in enumerate(result.get("contexts", [])):
                    context_text = context.get("context", "")
                    # Highlight figure references
                    context_text = re.sub(
                        r'(Fig(?:ure)?\.?\s*' + re.escape(figure_num) + r')', 
                        r'<strong style="background-color: yellow;">\1</strong>', 
                        context_text, 
                        flags=re.IGNORECASE
                    )
                    contexts_html += f'<div class="context-block"><strong>Reference {i+1}:</strong><br>{context_text}</div>'
                
                f.write(f"""
    <div class="figure">
        <h2>Figure {figure_num}</h2>
        
        <h3>Paper Context References</h3>
        {contexts_html}
        
        <h3>Raw Methodology Text Extracted from Paper</h3>
        <div class="method-text">
            {result.get("raw_methodology_text", "No methodology text found.")}
        </div>
        
        <h3>Reproduction Instructions</h3>
        <div class="instructions">
            {result.get("reproduction_instructions", "No instructions generated.")}
        </div>
    </div>
""")
            
            f.write("""
</body>
</html>""")
        
        print(f"Results saved to {self.output_dir} directory")

    def extract_text_around_figure(self, figure_num, context_size=1000):
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
    
    def _is_computational_methodology(self, figure_num, caption, contexts):
        """Determine if a figure involves computational methodology based on caption and context."""
        # Combine caption and context for analysis
        combined_text = caption + " " + " ".join([ctx["context"] for ctx in contexts[:3]])
        combined_text = combined_text.lower()
        
        # Keywords related to computational methodology
        comp_keywords = [
            'algorithm', 'computation', 'model', 'simulation', 'software', 
            'code', 'pipeline', 'processing', 'analysis', 'dataset',
            'neural network', 'machine learning', 'deep learning', 'statistical', 
            'parameter', 'optimization', 'regression', 'classification',
            'cluster', 'visualization', 'data processing', 'feature', 
            'training', 'prediction', 'validation', 'accuracy',
            'framework', 'implementation', 'performance', 'benchmark',
            'database', 'compute', 'gpu', 'cpu', 'memory', 'parallel'
        ]
        
        # Check if any computational keywords are in the text
        keyword_matches = [keyword for keyword in comp_keywords if keyword in combined_text]
        
        # Check if the figure seems to be a computational result
        is_computational = len(keyword_matches) >= 2  # At least 2 keywords to be considered computational
        
        # Print debug info
        if is_computational:
            matched_keywords = ", ".join(keyword_matches[:5])  # Show first 5 matches
            print(f"Figure {figure_num} identified as computational (keywords: {matched_keywords})")
        else:
            print(f"Figure {figure_num} does not appear to involve computational methodology - skipping")
            
        return is_computational, keyword_matches

    def extract_and_save_methodology(self, max_figures=10):
        """Extract methodology text for figures and save it without generating instructions."""
        # Find all figures
        all_figures = self._find_figures_in_text()
        print(f"Found {len(all_figures)} figures in the paper")
        
        # Extract figure captions
        captions = self._extract_figure_captions()
        print(f"Extracted {len(captions)} figure captions")
        
        # Filter figures based on computational methodology
        computational_figures = []
        for figure_num, contexts in all_figures.items():
            caption = captions.get(figure_num, "")
            is_computational, _ = self._is_computational_methodology(figure_num, caption, contexts)
            
            if is_computational:
                computational_figures.append((figure_num, contexts, caption))
                
        print(f"Identified {len(computational_figures)} figures with computational methodology")
        
        # Limit to max_figures
        figures_to_process = computational_figures[:max_figures]
        
        # Process each figure
        results = []
        for figure_num, contexts, caption in figures_to_process:
            print(f"Processing computational Figure {figure_num}...")
            
            # Extract methodology text for this figure
            methodology_text = self._extract_methods_for_figure(figure_num, contexts)
            
            # Extract text around the figure
            surrounding_text = self.extract_text_around_figure(figure_num, 2000)
            
            results.append({
                "figure_number": figure_num,
                "caption": caption,
                "raw_methodology_text": methodology_text,
                "contexts": contexts[:3],  # Include only first 3 contexts for brevity
                "surrounding_text": surrounding_text[:3]  # Include only first 3 surrounding texts
            })
        
        # Save JSON results
        with open(os.path.join(self.output_dir, 'computational_methodology.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        # Create HTML summary
        with open(os.path.join(self.output_dir, 'computational_methodology.html'), 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Computational Methodology Text for Figures</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .figure { 
            border: 1px solid #ddd; 
            padding: 20px; 
            margin-bottom: 30px; 
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .caption {
            background-color: #fffdeb;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #fbbc05;
            font-style: italic;
        }
        .method-text { 
            background-color: #f0f8ff; 
            padding: 15px; 
            margin: 15px 0;
            border-left: 4px solid #4285f4;
            white-space: pre-wrap;
            font-size: 0.9em;
        }
        .surrounding-text {
            background-color: #f5f5f5;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #ea4335;
            white-space: pre-wrap;
            font-size: 0.9em;
        }
        .context-block {
            background-color: #f9f9f9;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #0f9d58;
            font-size: 0.9em;
        }
        h1 { color: #4285f4; }
        h2 { color: #4285f4; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        h3 { color: #4285f4; }
    </style>
</head>
<body>
    <h1>Computational Methodology Text for Figures</h1>
    <p>This document contains methodology text extracted directly from the paper for figures involving computational methods.</p>
""")
            
            # Add each figure
            for result in results:
                figure_num = result["figure_number"]
                caption = result.get("caption", "No caption available")
                
                # Format context blocks
                contexts_html = ""
                for i, context in enumerate(result.get("contexts", [])):
                    context_text = context.get("context", "")
                    # Highlight figure references
                    context_text = re.sub(
                        r'(Fig(?:ure)?\.?\s*' + re.escape(figure_num) + r')', 
                        r'<strong style="background-color: yellow;">\1</strong>', 
                        context_text, 
                        flags=re.IGNORECASE
                    )
                    contexts_html += f'<div class="context-block"><strong>Reference {i+1}:</strong><br>{context_text}</div>'
                
                # Format surrounding text
                surrounding_html = ""
                for i, text in enumerate(result.get("surrounding_text", [])):
                    # Highlight figure references
                    highlighted_text = re.sub(
                        r'(Fig(?:ure)?\.?\s*' + re.escape(figure_num) + r')', 
                        r'<strong style="background-color: yellow;">\1</strong>', 
                        text, 
                        flags=re.IGNORECASE
                    )
                    surrounding_html += f'<div class="surrounding-text"><strong>Extended Context {i+1}:</strong><br>{highlighted_text}</div>'
                
                f.write(f"""
    <div class="figure">
        <h2>Figure {figure_num}</h2>
        
        <h3>Caption</h3>
        <div class="caption">
            {caption}
        </div>
        
        <h3>Paper Context References</h3>
        {contexts_html}
        
        <h3>Raw Methodology Text Extracted from Paper</h3>
        <div class="method-text">
            {result.get("raw_methodology_text", "No methodology text found.")}
        </div>
        
        <h3>Extended Surrounding Text</h3>
        {surrounding_html}
    </div>
""")
            
            f.write("""
</body>
</html>""")
        
        print(f"Computational methodology results saved to {self.output_dir} directory")
        return results

def main():
    try:
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not found in environment variables")
            print("Please create a .env file with your OpenAI API key")
            return
            
        # PDF path
        pdf_path = 'second.pdf'
        
        # Check if PDF exists
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return

        print("Initializing DirectPaperInstructions...")
        extractor = DirectPaperInstructions(pdf_path)
        
        # Extract methodology text only for computational figures
        print("Extracting methodology text for computational figures...")
        extractor.extract_and_save_methodology(max_figures=10)
        
        print("\nComputational methodology text has been extracted successfully!")
        print(f"- JSON file: {os.path.join(extractor.output_dir, 'computational_methodology.json')}")
        print(f"- HTML file: {os.path.join(extractor.output_dir, 'computational_methodology.html')}")
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 