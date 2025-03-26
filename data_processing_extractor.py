import os
import sys
import re
import json
import traceback
from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

class DataProcessingExtractor:
    def __init__(self, findings_json_path, pdf_path):
        self.findings_json_path = findings_json_path
        self.pdf_path = pdf_path
        self.client = None
        self.output_dir = 'data_processing_output'
        
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
        
        # Load findings JSON
        try:
            with open(self.findings_json_path, 'r') as f:
                self.findings_data = json.load(f)
            print(f"Loaded findings data with {len(self.findings_data.get('figure_references', []))} figure references")
        except Exception as e:
            print(f"Error loading findings JSON: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
            
        # Load PDF content
        try:
            self.pdf_content = self._extract_pdf_content()
            print(f"PDF content extracted. Length: {len(self.pdf_content)} characters")
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

    def _is_data_processing_related(self, finding):
        """Check if a finding is related to data processing."""
        # Keywords related to data processing
        data_processing_keywords = [
            'algorithm', 'computation', 'processing', 'analysis', 'statistical', 
            'model', 'neural network', 'machine learning', 'training', 'dataset',
            'filter', 'parameter', 'optimization', 'cluster', 'classification',
            'regression', 'transform', 'feature', 'pipeline', 'preprocessing',
            'normalization', 'validation', 'evaluation', 'metrics', 'accuracy',
            'data', 'database', 'compute', 'calculation', 'code', 'programming',
            'software', 'framework', 'algorithm', 'method', 'technique'
        ]
        
        # Check finding text
        finding_text = finding.get('finding', '') + ' ' + finding.get('context', '')
        finding_text = finding_text.lower()
        
        # Count keyword matches
        keyword_count = 0
        for keyword in data_processing_keywords:
            if keyword.lower() in finding_text:
                keyword_count += 1
                
        # Return score based on keyword density
        return {
            'is_data_related': keyword_count > 1,
            'data_relevance_score': keyword_count,
            'finding': finding
        }

    def _find_data_processing_section(self, finding):
        """Find the relevant section that explains data processing for a finding."""
        # Prepare search context
        finding_text = finding.get('finding', '')
        context = finding.get('context', '')
        figure_numbers = finding.get('figure_numbers', [])
        
        # Create a prompt to identify data processing details
        prompt = f"""
        I need to find the data processing details related to the following research finding:
        
        Finding: {finding_text}
        Context: {context}
        {"Related Figure(s): " + ", ".join(figure_numbers) if figure_numbers else ""}
        
        Please search through this section of the paper and identify any text that specifically explains:
        1. Data processing methods
        2. Algorithms or computational techniques
        3. Software or tools used for analysis
        4. Parameters or configurations for data processing
        5. Statistical methods applied to the data
        
        If there are multiple data processing steps, please identify all of them.
        
        Paper section:
        {{paper_section}}
        """
        
        # Find relevant sections in PDF content
        results = []
        # Generate search windows around figure references if available
        search_windows = []
        
        # If figure numbers are available, search around figure references
        if figure_numbers:
            for fig_num in figure_numbers:
                # Look for references to this figure in text
                pattern = r"(?:Fig(?:ure)?\.?\s*" + re.escape(fig_num) + r")"
                for match in re.finditer(pattern, self.pdf_content, re.IGNORECASE):
                    # Create window around match (3000 chars before and after)
                    start = max(0, match.start() - 3000)
                    end = min(len(self.pdf_content), match.end() + 3000)
                    search_windows.append(self.pdf_content[start:end])
        
        # If no figure-specific windows, use finding context to locate in text
        if not search_windows and context:
            # Try to find context in full text
            context_snippet = context[:100]  # Use first 100 chars of context
            if context_snippet in self.pdf_content:
                pos = self.pdf_content.find(context_snippet)
                # Create window around match (3000 chars before and after)
                start = max(0, pos - 3000)
                end = min(len(self.pdf_content), pos + 3000)
                search_windows.append(self.pdf_content[start:end])
        
        # If still no windows, use smaller chunks of the whole document
        if not search_windows:
            # Divide document into overlapping chunks
            chunk_size = 6000
            overlap = 1000
            for i in range(0, len(self.pdf_content), chunk_size - overlap):
                search_windows.append(self.pdf_content[i:i+chunk_size])
        
        # Process each search window
        for i, window in enumerate(search_windows):
            try:
                # Update prompt with this section of text
                current_prompt = prompt.replace('{paper_section}', window)
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a scientific paper analyzer that specializes in identifying and extracting data processing methods from research papers."},
                        {"role": "user", "content": current_prompt}
                    ],
                    temperature=0.1
                )
                
                content = response.choices[0].message.content
                
                # Only add if actual content was found
                if 'no data processing' not in content.lower() and len(content) > 50:
                    results.append({
                        'finding': finding,
                        'data_processing_details': content,
                        'window_index': i
                    })
                    
                    # If we found good details, don't need to check other windows
                    if len(content) > 200:
                        break
                    
            except Exception as e:
                print(f"Error analyzing section {i}: {str(e)}")
                continue
                
        return results

    def extract_data_processing_findings(self, max_findings=5):
        """Extract findings related to data processing and find their explanations."""
        all_findings = []
        
        # Process each section's findings
        for section, findings in self.findings_data.get('findings_by_section', {}).items():
            for finding in findings:
                result = self._is_data_processing_related(finding)
                if result['is_data_related']:
                    result['section'] = section
                    all_findings.append(result)
        
        # Sort by relevance score
        all_findings.sort(key=lambda x: x['data_relevance_score'], reverse=True)
        
        # Limit to max_findings
        top_findings = all_findings[:max_findings]
        print(f"Found {len(top_findings)} data processing related findings out of {len(all_findings)} total potential matches")
        
        # Get detailed data processing sections for each finding
        detailed_findings = []
        for finding_info in top_findings:
            finding = finding_info['finding']
            section = finding_info['section']
            print(f"Processing finding: {finding.get('finding', '')[:50]}...")
            
            # Get data processing details
            details = self._find_data_processing_section(finding)
            if details:
                for detail in details:
                    detail['section'] = section
                    detail['data_relevance_score'] = finding_info['data_relevance_score']
                    detailed_findings.append(detail)
        
        return detailed_findings

    def generate_output(self, detailed_findings):
        """Generate HTML and JSON output for the detailed findings."""
        # Save JSON results
        with open(os.path.join(self.output_dir, 'data_processing_details.json'), 'w') as f:
            json.dump(detailed_findings, f, indent=2)
        
        # Create HTML summary
        with open(os.path.join(self.output_dir, 'data_processing_summary.html'), 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Data Processing Details from Research Paper</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .finding { 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin-bottom: 20px; 
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .data-section { 
            background-color: #e8f4fe; 
            padding: 15px; 
            margin-top: 10px;
            border-left: 4px solid #4285f4;
        }
        .figure-ref { 
            color: #0f9d58;
            font-weight: bold;
        }
        h1 { color: #4285f4; }
        h2 { color: #4285f4; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        h3 { color: #4285f4; }
        pre { white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>Data Processing Details from Research Paper</h1>
    <p>This document contains the 5 most relevant data processing findings from the paper and their detailed explanations.</p>
""")
            
            # Add each finding
            for i, detail in enumerate(detailed_findings):
                finding = detail['finding']
                section = detail.get('section', 'Unknown Section')
                relevance_score = detail.get('data_relevance_score', 0)
                
                figure_info = ""
                if finding.get("figure_numbers", []):
                    figure_info = f"<div class='figure-ref'>References Figure(s): {', '.join(finding.get('figure_numbers', []))}</div>"
                
                f.write(f"""
    <div class="finding">
        <h2>Finding {i+1}: {section}</h2>
        <h3>{finding.get("finding", "")}</h3>
        {figure_info}
        <p><em>Context:</em> {finding.get("context", "")}</p>
        <p><em>Data Relevance Score:</em> {relevance_score}</p>
        
        <div class="data-section">
            <h3>Data Processing Details:</h3>
            <pre>{detail.get("data_processing_details", "No details found")}</pre>
        </div>
    </div>
""")
            
            f.write("""
</body>
</html>""")
        
        print(f"Results saved to {self.output_dir} directory")

def main():
    try:
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not found in environment variables")
            print("Please create a .env file with your OpenAI API key")
            return
            
        # Define paths to input files
        findings_json_path = 'paper_findings_output/findings_results.json'
        pdf_path = 'pdf.pdf'
        
        # Check if findings JSON exists
        if not os.path.exists(findings_json_path):
            print(f"Error: Findings JSON file not found at {findings_json_path}")
            print("Please run paper_findings_extractor.py first to generate the findings JSON")
            return

        print("Initializing DataProcessingExtractor...")
        extractor = DataProcessingExtractor(findings_json_path, pdf_path)
        
        print("Extracting data processing findings...")
        detailed_findings = extractor.extract_data_processing_findings(max_findings=5)
        
        if not detailed_findings:
            print("No data processing details found in the paper")
            return
            
        print(f"Found detailed data processing information for {len(detailed_findings)} findings")
        
        # Generate output
        extractor.generate_output(detailed_findings)
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 