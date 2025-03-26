#!/usr/bin/env python3
import os
import json
import fitz  # PyMuPDF
import re
import logging
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PaperAnalyzer:
    def __init__(self, pdf_path, output_dir="paper_analysis"):
        """Initialize the paper analyzer with a PDF path and output directory"""
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.ensure_output_dirs()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.full_text = ""
        self.metadata = {}
        self.findings = []
        self.computational_findings = []
        
    def ensure_output_dirs(self):
        """Create output directories if they don't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "findings"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "computational_findings"), exist_ok=True)
        logger.info(f"Output directories created in {self.output_dir}")
        
    def extract_text_from_pdf(self):
        """Extract full text from PDF"""
        logger.info(f"Extracting text from {self.pdf_path}")
        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for page_num in tqdm(range(len(doc)), desc="Reading pages"):
                page = doc[page_num]
                text += page.get_text()
            self.full_text = text
            
            # Save full text to file
            full_text_path = os.path.join(self.output_dir, "full_text.txt")
            with open(full_text_path, "w", encoding="utf-8") as f:
                f.write(self.full_text)
            
            logger.info(f"Full text saved to {full_text_path}")
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def chunk_text(self, text, max_chunk_size=4000):
        """Split text into chunks of approximately max_chunk_size characters"""
        chunks = []
        current_chunk = ""
        
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                chunks.append(current_chunk)
                current_chunk = para + "\n\n"
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def extract_metadata(self):
        """Extract metadata from the first few pages"""
        logger.info("Extracting paper metadata")
        
        # Extract first ~2000 characters for metadata extraction
        metadata_text = self.full_text[:8000]
        
        # Template for metadata extraction
        prompt = """
        Analyze this scientific paper and extract the following metadata in a structured format:
        1. Title
        2. Authors (with affiliations if available)
        3. Journal/Conference name
        4. Publication date
        5. DOI/Identifier
        6. Abstract (complete text)
        7. Keywords
        8. Main research areas/fields
        9. Corresponding author contact information

        Format the extracted information as a JSON object.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a scientific paper metadata extractor."},
                    {"role": "user", "content": f"{prompt}\n\nPaper text:\n{metadata_text}"}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            metadata_text = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                # Extract JSON if it's embedded in markdown or other text
                json_match = re.search(r'```json\n(.*?)\n```', metadata_text, re.DOTALL)
                if json_match:
                    metadata_json = json_match.group(1)
                else:
                    metadata_json = metadata_text
                
                self.metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                # If JSON parsing fails, save as plain text
                logger.warning("Could not parse metadata as JSON, saving as text")
                self.metadata = {"raw_metadata": metadata_text}
            
            # Save metadata to file
            metadata_path = os.path.join(self.output_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {metadata_path}")
            return self.metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            self.metadata = {"error": str(e)}
            return self.metadata
    
    def identify_findings(self):
        """Identify main findings from the paper"""
        logger.info("Identifying main findings")
        
        chunks = self.chunk_text(self.full_text)
        
        # Template for finding identification
        prompt = """
        Identify the main computational findings from this scientific paper. Focus on findings that:

        1. Involve computational methods, algorithms, or data analysis
        2. Describe a novel computational approach, model, or technique
        3. Present results from computational experiments or simulations
        4. Use statistical analysis, machine learning, or data mining
        5. Involve software tools, code implementation, or computational pipelines

        For each finding:
        - Provide a concise one-sentence description of the computational result
        - Include the section and approximate location where it appears
        - Note any related figures, tables, algorithms, or equations
        - Identify which computational method or analysis produced this finding

        Format the findings as a JSON array of objects with these properties:
        - finding_id
        - description (focus on the computational aspect)
        - location (section and page if available)
        - related_figures
        - related_tables
        - computational_methods (list of methods, algorithms, or tools used)
        - is_computational (should be true for all findings)

        Identify at least 5 computational findings, prioritizing those with clear methodology descriptions that could be reproduced with the right data.
        """
        
        all_findings = []
        
        try:
            # Process each chunk to identify findings
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} for findings")
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a scientific research analyst specializing in extracting computational methods and findings from papers. Focus on findings that involve algorithms, data processing, statistical analysis, or computational techniques."},
                        {"role": "user", "content": f"{prompt}\n\nPaper text (chunk {i+1}/{len(chunks)}):\n{chunk}"}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
                
                findings_text = response.choices[0].message.content
                
                # Try to parse as JSON
                try:
                    # Extract JSON if it's embedded in markdown or other text
                    json_match = re.search(r'```json\n(.*?)\n```', findings_text, re.DOTALL)
                    if json_match:
                        findings_json = json_match.group(1)
                    else:
                        findings_json = findings_text
                    
                    chunk_findings = json.loads(findings_json)
                    if isinstance(chunk_findings, list):
                        all_findings.extend(chunk_findings)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse findings from chunk {i+1} as JSON")
                    continue
            
            # Deduplicate findings
            unique_findings = []
            descriptions = set()
            
            for finding in all_findings:
                desc = finding.get("description", "")
                if desc and desc not in descriptions:
                    descriptions.add(desc)
                    # Make sure is_computational is set to true
                    finding["is_computational"] = True
                    unique_findings.append(finding)
            
            self.findings = unique_findings
            
            # Save findings to file
            findings_path = os.path.join(self.output_dir, "findings.json")
            with open(findings_path, "w", encoding="utf-8") as f:
                json.dump(self.findings, f, indent=2)
            
            logger.info(f"Found {len(self.findings)} unique computational findings, saved to {findings_path}")
            
            # All findings are computational now, so just take the top 5
            self.computational_findings = self.findings[:5]
            logger.info(f"Selected top {len(self.computational_findings)} computational findings for detailed analysis")
            
            return self.findings
            
        except Exception as e:
            logger.error(f"Error identifying findings: {str(e)}")
            return []
    
    def analyze_computational_finding(self, finding):
        """Analyze a computational finding in detail"""
        finding_id = finding.get("finding_id", "unknown")
        description = finding.get("description", "")
        
        logger.info(f"Analyzing computational finding: {finding_id}")
        
        # Template for computational finding analysis
        prompt = f"""
        Focus on Finding #{finding_id} in this paper about {description}.

        1. IDENTIFICATION:
           - Locate and extract all text, figures, tables, and references related to this finding
           - Identify where in the paper this finding is mentioned (sections, page numbers)
           - Extract the exact statements that describe this finding

        2. METHODOLOGY EXTRACTION (DETAILED):
           - Detail all methods, techniques, and protocols used to produce this finding
           - List all equipment, software packages, libraries, and tools with version numbers where available
           - Extract all parameters, settings, configurations, and hyperparameters
           - Identify all statistical methods, algorithms, and analyses applied
           - Extract any mathematical formulas or equations relevant to the finding
           - Describe any pre-processing or normalization steps

        3. IMPLEMENTATION DETAILS:
           - Identify the exact implementation steps in sequential order
           - Extract any pseudocode, algorithms, or code snippets mentioned
           - Detail specific functions or methods described in the paper
           - Note any custom implementations or modifications to existing methods
           - Describe any parallel processing, distributed computing, or optimization techniques

        4. REPRODUCTION PROCEDURE:
           - Provide a detailed step-by-step procedure to reproduce this finding
           - Describe exactly how the data should be processed (assume the data is already available)
           - Specify all computational requirements (hardware, software, memory, processing time)
           - Detail any critical implementation details that might affect results
           - List all validation and verification steps to confirm correct implementation

        5. EXPECTED RESULTS:
           - Describe the expected output format and structure
           - Detail key metrics, measurements, and evaluation criteria
           - Specify expected ranges for results based on paper findings
           - Note any baseline comparisons that should be reproduced
           - Describe visualizations or analyses needed to verify results

        6. VALIDATION AND TROUBLESHOOTING:
           - Explain how the authors validated this finding
           - List potential issues that might arise during reproduction
           - Provide troubleshooting steps for common problems
           - Describe signs of successful implementation vs. signs of errors
           - Note any sensitivity analyses or robustness checks

        Organize this information in a clear, structured format with emphasis on the exact steps needed to reproduce the finding, assuming the dataset is already available.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a scientific computational methodology expert. Your task is to extract detailed methodology and reproduction steps from research papers, with particular attention to implementation details and exactly how to use the existing data."},
                    {"role": "user", "content": f"{prompt}\n\nPaper text:\n{self.full_text}"}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            
            analysis = response.choices[0].message.content
            
            # Save analysis to file
            filename = f"finding_{finding_id}_analysis.txt"
            filepath = os.path.join(self.output_dir, "computational_findings", filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Analysis of Finding {finding_id}\n\n")
                f.write(f"Description: {description}\n\n")
                f.write(analysis)
            
            logger.info(f"Analysis of finding {finding_id} saved to {filepath}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing finding {finding_id}: {str(e)}")
            return ""
    
    def process_paper(self):
        """Process the paper: extract text, metadata, findings, and analyze computational findings"""
        try:
            # Extract full text
            self.extract_text_from_pdf()
            
            # Extract metadata
            self.extract_metadata()
            
            # Identify findings
            self.identify_findings()
            
            # Analyze top 5 computational findings
            for finding in self.computational_findings:
                self.analyze_computational_finding(finding)
            
            # Generate summary report
            self.generate_summary_report()
            
            logger.info("Paper analysis complete")
            
        except Exception as e:
            logger.error(f"Error processing paper: {str(e)}")
    
    def generate_summary_report(self):
        """Generate a summary report of the analysis"""
        summary = {
            "metadata": self.metadata,
            "total_findings": len(self.findings),
            "computational_findings": len(self.computational_findings),
            "computational_finding_ids": [f.get("finding_id") for f in self.computational_findings]
        }
        
        summary_path = os.path.join(self.output_dir, "summary_report.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a scientific paper PDF")
    parser.add_argument("pdf_path", help="Path to the PDF file to analyze")
    parser.add_argument("-o", "--output", dest="output_dir", default="paper_analysis", 
                        help="Directory to store analysis results (default: paper_analysis)")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        return 1
    
    analyzer = PaperAnalyzer(args.pdf_path, args.output_dir)
    analyzer.process_paper()
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}/")
    return 0

if __name__ == "__main__":
    exit(main()) 