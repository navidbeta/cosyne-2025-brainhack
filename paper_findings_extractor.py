import os
import sys
import re
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import json
import traceback

# Load environment variables
load_dotenv()

class PaperFindingsExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.client = None
        self.output_dir = 'paper_findings_output'
        
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
        
        try:
            self.content = self._extract_pdf_content()
            print(f"PDF content extracted. Length: {len(self.content)} characters")
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

    def _identify_sections(self):
        """Identify main sections of the paper."""
        prompt = f"""
        Analyze the following academic paper and identify its main sections (like Abstract, Introduction, Methods, Results, Discussion, etc.).
        For each section, provide the section title and the approximate start of that section in the text.
        
        Format your response as a JSON object with section names as keys and their approximate starting position as values:
        
        {{
            "Abstract": 0,
            "Introduction": 500,
            "Methods": 1200,
            ...
        }}
        
        Paper text:
        {self.content[:5000]}...
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
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
        return chunks

    def extract_findings_and_figure_references(self):
        """Extract key findings and figure references from the paper."""
        chunks = self._chunk_content()
        sections = self._identify_sections()
        
        findings = []
        figure_references = []
        
        for i, chunk in enumerate(chunks):
            print(f"Analyzing chunk {i+1}/{len(chunks)}")
            
            # Find figure references in this chunk
            refs = self._find_figure_references(chunk)
            for ref in refs:
                ref["chunk_index"] = i
                figure_references.append(ref)
            
            # Extract findings from this chunk
            chunk_findings = self._extract_findings_from_chunk(chunk, i)
            findings.extend(chunk_findings)
        
        # Organize findings by section
        organized_findings = self._organize_by_section(findings, sections)
        
        # Map figure references to findings where possible
        findings_with_figures = self._map_figures_to_findings(organized_findings, figure_references)
        
        return {
            "findings_by_section": findings_with_figures,
            "figure_references": figure_references
        }
    
    def _find_figure_references(self, text):
        """Find references to figures in the text."""
        # Pattern to find references like "Figure 1", "Fig. 2A", etc.
        figure_pattern = r"(Fig(?:ure)?\.?\s*(\d+[A-Za-z]?))"
        matches = re.finditer(figure_pattern, text, re.IGNORECASE)
        
        references = []
        for match in matches:
            full_ref = match.group(1)
            fig_num = match.group(2)
            
            # Get context around the reference (100 chars before and after)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            
            references.append({
                "figure_number": fig_num,
                "full_reference": full_ref,
                "reference_context": context,
                "position_in_chunk": match.start()
            })
        
        return references
    
    def _extract_findings_from_chunk(self, chunk, chunk_index):
        """Extract key findings from a chunk of text."""
        prompt = f"""
        Analyze this section of an academic paper and extract all key findings, results, or conclusions.
        
        For each finding:
        1. State the finding clearly and concisely
        2. Note if it references any figures/tables
        3. Indicate the relative importance of the finding (major or minor)
        
        Format your response as a JSON array where each finding is an object with the following structure:
        
        {{
            "findings": [
                {{
                    "finding": "Finding statement",
                    "references_figure": true/false,
                    "figure_numbers": ["1", "2A", etc.],
                    "importance": "major/minor",
                    "context": "Brief surrounding context of the finding"
                }}
            ]
        }}
        
        Text chunk:
        {chunk[:3000]}...
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a scientific findings extractor that identifies key results and conclusions from research papers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            try:
                # Extract JSON from the response
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    findings_json = json.loads(json_match.group(1))
                    findings = findings_json.get("findings", [])
                    
                    # Add chunk index to each finding
                    for finding in findings:
                        finding["chunk_index"] = chunk_index
                    
                    return findings
                else:
                    print("Could not extract findings JSON from response")
                    return []
            except Exception as e:
                print(f"Error parsing findings JSON: {str(e)}")
                return []
                
        except Exception as e:
            print(f"Error extracting findings: {str(e)}")
            return []
    
    def _organize_by_section(self, findings, sections):
        """Organize findings by paper section."""
        if not sections:
            return {"Unsorted": findings}
        
        organized = {}
        section_starts = list(sections.items())
        section_starts.sort(key=lambda x: x[1])  # Sort by start position
        
        for i, (section_name, start_pos) in enumerate(section_starts):
            # Determine section end position
            if i < len(section_starts) - 1:
                end_pos = section_starts[i+1][1]
            else:
                end_pos = float('inf')
            
            # Filter findings that belong to this section
            section_findings = []
            for finding in findings:
                # Estimate position of finding in document
                chunk_index = finding.get("chunk_index", 0)
                approx_position = chunk_index * 8000  # Rough estimation
                
                if start_pos <= approx_position < end_pos:
                    section_findings.append(finding)
            
            organized[section_name] = section_findings
        
        # Add any unmatched findings to "Other" section
        all_matched = sum(len(f) for f in organized.values())
        if all_matched < len(findings):
            organized["Other"] = [f for f in findings if not any(f in section_findings for section_findings in organized.values())]
        
        return organized
    
    def _map_figures_to_findings(self, organized_findings, figure_references):
        """Map figure references to relevant findings."""
        # Create a dictionary of figure numbers to their references
        figure_dict = {}
        for ref in figure_references:
            fig_num = ref["figure_number"]
            if fig_num not in figure_dict:
                figure_dict[fig_num] = []
            figure_dict[fig_num].append(ref)
        
        # For each section and finding, see if we can match figure references
        for section, findings in organized_findings.items():
            for finding in findings:
                # If finding already has figure references, skip
                if finding.get("references_figure", False) and finding.get("figure_numbers", []):
                    continue
                    
                # Check if finding text mentions figures
                finding_text = finding.get("finding", "") + finding.get("context", "")
                mentioned_figures = []
                
                for fig_num in figure_dict:
                    if f"Fig {fig_num}" in finding_text or f"Figure {fig_num}" in finding_text:
                        mentioned_figures.append(fig_num)
                
                if mentioned_figures:
                    finding["references_figure"] = True
                    finding["figure_numbers"] = mentioned_figures
        
        return organized_findings

def main():
    try:
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not found in environment variables")
            print("Please create a .env file with your OpenAI API key")
            return

        print("Initializing PaperFindingsExtractor...")
        extractor = PaperFindingsExtractor('pdf.pdf')
        
        print("Extracting findings and figure references...")
        results = extractor.extract_findings_and_figure_references()
        
        if not results.get("findings_by_section"):
            print("No findings obtained from analysis")
            return
            
        # Save results to a JSON file
        with open(os.path.join(extractor.output_dir, 'findings_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {os.path.join(extractor.output_dir, 'findings_results.json')}")
        
        # Create a summary HTML file for easy viewing
        with open(os.path.join(extractor.output_dir, 'findings_summary.html'), 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Paper Findings Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin-bottom: 20px; 
            border-radius: 5px;
            background-color: #f9f9f9;
        }}
        .finding {{ 
            border-left: 4px solid #4285f4; 
            padding: 10px; 
            margin: 10px 0;
            background-color: white;
        }}
        .major {{ border-left-color: #ea4335; }}
        .figure-ref {{ 
            color: #0f9d58;
            font-weight: bold;
        }}
        h1 {{ color: #4285f4; }}
        h2 {{ color: #4285f4; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        h3 {{ color: #4285f4; }}
    </style>
</head>
<body>
    <h1>Paper Findings: {os.path.basename(extractor.pdf_path)}</h1>
    
    <h2>Findings by Section</h2>""")
            
            # Generate sections HTML
            for section_name, findings in results["findings_by_section"].items():
                if not findings:
                    continue
                    
                f.write(f"""
    <div class="section">
        <h3>{section_name} ({len(findings)} findings)</h3>""")
                
                # Generate findings HTML
                for finding in findings:
                    importance_class = "major" if finding.get("importance") == "major" else ""
                    figure_info = ""
                    if finding.get("references_figure", False) and finding.get("figure_numbers", []):
                        figure_info = f"<div class='figure-ref'>References Figure(s): {', '.join(finding.get('figure_numbers', []))}</div>"
                    
                    f.write(f"""
        <div class="finding {importance_class}">
            <p><strong>{finding.get("finding", "")}</strong></p>
            {figure_info}
            <p><em>Context:</em> {finding.get("context", "")}</p>
        </div>""")
                
                f.write("""
    </div>""")
            
            f.write("""
    
    <h2>Figure References</h2>
    <div class="section">""")
            
            # Organize figure references by figure number
            figures_organized = {}
            for ref in results["figure_references"]:
                fig_num = ref["figure_number"]
                if fig_num not in figures_organized:
                    figures_organized[fig_num] = []
                figures_organized[fig_num].append(ref)
            
            # Sort by figure number
            for fig_num in sorted(figures_organized.keys(), key=lambda x: (int(re.sub(r'[^0-9]', '', x)), x)):
                refs = figures_organized[fig_num]
                f.write(f"""
        <h3>Figure {fig_num} ({len(refs)} references)</h3>""")
                
                for ref in refs:
                    context = ref.get("reference_context", "").replace(ref.get("full_reference", ""), f"<span style='background-color: yellow;'>{ref.get('full_reference', '')}</span>")
                    f.write(f"""
        <div class="finding">
            <p>{context}</p>
        </div>""")
            
            f.write("""
    </div>
</body>
</html>""")
        
        print(f"Summary HTML created at {os.path.join(extractor.output_dir, 'findings_summary.html')}")
        
        # Print a summary
        print("\nFindings Summary:")
        for section, section_findings in results["findings_by_section"].items():
            print(f"\n{section} ({len(section_findings)} findings)")
            major_findings = [f for f in section_findings if f.get("importance") == "major"]
            if major_findings:
                print(f"Major findings ({len(major_findings)}):")
                for finding in major_findings[:3]:  # Show only first 3
                    print(f"- {finding.get('finding', '')[:100]}...")
        
        print(f"\nFigure References: {len(results['figure_references'])} total references")
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 