import os
import sys
import json
import re
import traceback
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ReproductionInstructionsGenerator:
    def __init__(self, data_processing_json_path):
        self.data_processing_json_path = data_processing_json_path
        self.client = None
        self.output_dir = 'reproduction_instructions'
        
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
        
        # Load data processing details
        try:
            with open(self.data_processing_json_path, 'r') as f:
                self.processing_details = json.load(f)
            print(f"Loaded data processing details for {len(self.processing_details)} findings")
        except Exception as e:
            print(f"Error loading data processing JSON: {str(e)}")
            traceback.print_exc()
            sys.exit(1)

    def generate_reproduction_instructions(self):
        """Generate step-by-step instructions to reproduce each figure/result."""
        reproduction_instructions = []
        
        for i, detail in enumerate(self.processing_details):
            finding = detail.get('finding', {})
            finding_text = finding.get('finding', '')
            data_processing_details = detail.get('data_processing_details', '')
            figure_numbers = finding.get('figure_numbers', [])
            
            print(f"Generating reproduction instructions for finding {i+1}: {finding_text[:50]}...")
            
            # Create a prompt to generate reproduction instructions
            prompt = f"""
I need detailed, step-by-step instructions to reproduce the scientific figure(s) or result(s) described below.
Based on the data processing details provided, create a clear set of reproducible instructions that anyone could follow.

FINDING: {finding_text}

DATA PROCESSING DETAILS:
{data_processing_details}

{"RELATED FIGURE(S): " + ", ".join(figure_numbers) if figure_numbers else ""}

Please provide:
1. A complete list of required software, libraries, packages, and versions needed
2. Any datasets required, with information on how to obtain them
3. Detailed preprocessing steps, exactly as described in the methodology
4. All algorithm parameters, statistical methods, and computational procedures
5. Step-by-step code instructions (with pseudo-code or actual code snippets where possible)
6. Visualization/figure generation instructions
7. Expected output and how to validate the results

Format your response as a numbered list of steps, with clear headers for each major section.
Include code blocks where appropriate using markdown syntax.
"""
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",  # Using GPT-4 for higher quality instructions
                    messages=[
                        {"role": "system", "content": "You are a scientific reproduction expert that creates detailed instructions for reproducing scientific figures and results from research papers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )
                
                instructions_text = response.choices[0].message.content
                
                reproduction_instructions.append({
                    'finding': finding,
                    'data_processing_details': data_processing_details,
                    'reproduction_instructions': instructions_text,
                    'index': i + 1
                })
                
            except Exception as e:
                print(f"Error generating reproduction instructions: {str(e)}")
                traceback.print_exc()
                continue
        
        return reproduction_instructions

    def generate_output(self, reproduction_instructions):
        """Generate HTML and JSON output for the reproduction instructions."""
        # Save JSON results
        with open(os.path.join(self.output_dir, 'reproduction_instructions.json'), 'w') as f:
            json.dump(reproduction_instructions, f, indent=2)
        
        # Create HTML summary
        with open(os.path.join(self.output_dir, 'reproduction_instructions.html'), 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Reproduction Instructions for Scientific Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .finding { 
            border: 1px solid #ddd; 
            padding: 20px; 
            margin-bottom: 30px; 
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .instructions { 
            background-color: #fff; 
            padding: 20px; 
            margin-top: 15px;
            border-left: 4px solid #4285f4;
            white-space: pre-wrap;
        }
        .figure-ref { 
            color: #0f9d58;
            font-weight: bold;
            margin: 10px 0;
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
        .code-block {
            background-color: #f6f8fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: monospace;
            white-space: pre;
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <h1>Reproduction Instructions for Scientific Results</h1>
    <p>This document contains step-by-step instructions for reproducing the figures and results from the paper, based on the data processing methods described.</p>
""")
            
            # Process each finding
            for instr in reproduction_instructions:
                finding = instr.get('finding', {})
                finding_text = finding.get('finding', '')
                index = instr.get('index', 0)
                figure_numbers = finding.get('figure_numbers', [])
                
                figure_info = ""
                if figure_numbers:
                    figure_info = f"<div class='figure-ref'>Figure(s): {', '.join(figure_numbers)}</div>"
                
                # Process markdown-style code blocks in the instructions
                instructions_text = instr.get('reproduction_instructions', '')
                
                # Convert markdown code blocks to HTML
                instructions_html = ""
                in_code_block = False
                code_content = ""
                
                for line in instructions_text.split('\n'):
                    if line.startswith('```'):
                        if in_code_block:
                            # End of code block
                            instructions_html += f'<div class="code-block">{code_content}</div>'
                            code_content = ""
                            in_code_block = False
                        else:
                            # Start of code block
                            in_code_block = True
                    elif in_code_block:
                        code_content += line + "\n"
                    else:
                        # Special handling for inline code using backticks
                        line = re.sub(r'`([^`]+)`', r'<code>\1</code>', line)
                        instructions_html += line + "\n"
                
                # Add any remaining code block
                if in_code_block and code_content:
                    instructions_html += f'<div class="code-block">{code_content}</div>'
                
                f.write(f"""
    <div class="finding">
        <h2>Reproduction Instructions {index}</h2>
        <h3>{finding_text}</h3>
        {figure_info}
        
        <div class="instructions">
            {instructions_html}
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
            
        # Define path to input files
        data_processing_json_path = 'data_processing_output/data_processing_details.json'
        
        # Check if data processing JSON exists
        if not os.path.exists(data_processing_json_path):
            print(f"Error: Data processing JSON file not found at {data_processing_json_path}")
            print("Please run data_processing_extractor.py first to generate the data processing details")
            return

        print("Initializing ReproductionInstructionsGenerator...")
        generator = ReproductionInstructionsGenerator(data_processing_json_path)
        
        print("Generating reproduction instructions...")
        reproduction_instructions = generator.generate_reproduction_instructions()
        
        if not reproduction_instructions:
            print("No reproduction instructions generated")
            return
            
        print(f"Generated reproduction instructions for {len(reproduction_instructions)} findings")
        
        # Generate output
        generator.generate_output(reproduction_instructions)
        
        print("\nReproduction instructions have been generated successfully!")
        print(f"- JSON file: {os.path.join(generator.output_dir, 'reproduction_instructions.json')}")
        print(f"- HTML file: {os.path.join(generator.output_dir, 'reproduction_instructions.html')}")
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 