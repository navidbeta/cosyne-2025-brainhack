import os
import sys
import json
import traceback
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PaperQueryEngine:
    def __init__(self, content_json_path):
        self.content_json_path = content_json_path
        self.client = None
        self.content = None
        
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
        
        # Load content
        self.load_content()
    
    def load_content(self):
        """Load the extracted paper content from JSON."""
        try:
            with open(self.content_json_path, 'r') as f:
                self.content = json.load(f)
            print(f"Loaded content with {len(self.content.get('figures', {}))} figures")
        except Exception as e:
            print(f"Error loading content JSON: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
    
    def query_paper(self, query):
        """Query the paper content using the OpenAI API."""
        # Create a summary of the paper for context
        paper_summary = f"""This is a scientific paper with the following characteristics:
- {len(self.content.get('figures', {}))} figures identified
- {len(self.content.get('captions', {}))} figure captions extracted
- Full text length: {len(self.content.get('full_text', ''))} characters
"""
        
        # Construct the system message
        system_message = f"""You are a scientific research assistant analyzing a paper.
Your task is to provide accurate, detailed answers based only on the content of this paper.
If the information isn't present in the paper's content, acknowledge that limitation.
{paper_summary}
"""
        
        # Create figure-specific context
        figure_info = ""
        for fig_num, fig_data in self.content.get('figures', {}).items():
            figure_info += f"\nFigure {fig_num} Caption: {fig_data.get('caption', 'No caption available')}\n"
            # Add first reference context
            if fig_data.get('references') and len(fig_data.get('references')) > 0:
                figure_info += f"Context around Figure {fig_num}: {fig_data['references'][0].get('context', '')[:500]}...\n"
        
        # Create a prompt
        prompt = f"""
Based on the paper content, answer the following question:

{query}

Here are some figure captions and related context to help:
{figure_info[:4000]}

Remember to only use information that is present in the provided paper content.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using the most capable model for scientific analysis
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=3000  # Allow a substantial response
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying OpenAI API: {str(e)}")
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def query_computational_results(self):
        """Specifically query for the 5 most important computational results."""
        
        # Get full text for analysis
        full_text = self.content.get('full_text', '')
        
        # Create a prompt specifically for computational results
        query = """List the 5 most important computational results of this paper, and for each result:
1. Describe the result and its significance
2. Explain how the data was processed to obtain this result
3. Include direct quotes from the paper showing the methodology

Format your response with clear headings for each result and separate sections for the methodology.
Focus only on computational aspects like algorithms, data processing, statistical methods, etc.
"""
        
        # Instead of pre-filtering for method sections, provide a representative sample
        # of the full text (due to token limitations) and let GPT identify methods
        
        # Take the first and last portions of the paper to get intro and conclusion
        intro_text = full_text[:5000] if len(full_text) > 5000 else full_text
        conclusion_text = full_text[-5000:] if len(full_text) > 10000 else ""
        
        # Enhanced prompt with paper content
        enhanced_prompt = f"""
Based on the paper content, answer the following question:

{query}

Here is the beginning portion of the paper that may include the introduction and methods:
{intro_text}

Here is the end portion of the paper that may include results, discussion and conclusion:
{conclusion_text}

Here are some figure captions that may relate to computational results:
"""
        
        # Add all figure captions without filtering
        for fig_num, caption in self.content.get('captions', {}).items():
            enhanced_prompt += f"\nFigure {fig_num}: {caption}\n"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # Using a compatible model
                messages=[
                    {"role": "system", "content": "You are a computational science expert analyzing a research paper. Your task is to identify and explain computational methods and results accurately based solely on the paper's content. You need to extract this information directly from the paper without relying on pre-filtered methodology sections."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=3000  # Allow a substantial response
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying OpenAI API: {str(e)}")
            traceback.print_exc()
            return f"Error: {str(e)}"

    def query_datasets(self):
        """Query specifically about what datasets were used in the paper."""
        # Get full text for analysis
        full_text = self.content.get('full_text', '')
        
        # Create a prompt specifically for dataset identification
        query = """Based on the paper content, please identify:

1. What specific datasets were used in this study
2. How these datasets were processed or prepared
3. Any notable characteristics of the data (size, features, etc.)
4. How the datasets were applied to the computational methods

Please include direct quotes from the paper where possible and be specific about the datasets used.
"""
        
        # Instead of pre-filtering for data sections, provide samples from different
        # parts of the paper to get a representative view
        beginning = full_text[:3000] if len(full_text) > 3000 else full_text
        middle_start = len(full_text) // 3 if len(full_text) > 9000 else 0
        middle = full_text[middle_start:middle_start+3000] if middle_start > 0 else ""
        end_start = len(full_text) // 3 * 2 if len(full_text) > 9000 else 0
        end = full_text[end_start:end_start+3000] if end_start > 0 else ""
        
        # Enhanced prompt with paper samples
        enhanced_prompt = f"""
Based on the paper content, answer the following question:

{query}

Here are sections from the paper (beginning, middle, and end):

BEGINNING:
{beginning}

MIDDLE:
{middle}

END:
{end}

Here are some figure captions that may relate to datasets:
"""
        
        # Add all figure captions without filtering
        for fig_num, caption in self.content.get('captions', {}).items():
            enhanced_prompt += f"\nFigure {fig_num}: {caption}\n"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a compatible model
                messages=[
                    {"role": "system", "content": "You are a data science expert analyzing a research paper. Your task is to identify and explain what datasets were used in the paper and how they were processed. You should extract this information directly from the paper content without relying on pre-filtered sections."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=2000  # Shorter response for dataset information
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying OpenAI API: {str(e)}")
            traceback.print_exc()
            return f"Error: {str(e)}"

    def generate_figure_code_instructions(self):
        """Generate code instructions for recreating similar figures from the paper."""
        # Get figure captions and references
        figures = self.content.get('figures', {})
        captions = self.content.get('captions', {})
        
        # Use all figures instead of filtering by visualization keywords
        viz_figures = figures
        
        # Construct the prompt for generating code instructions
        prompt = """Based on the figure captions and surrounding text context, create Python code instructions for recreating similar figures from the paper. 

For each figure, provide:
1. An explanation of what the figure is showing and its significance
2. Instructions for data preparation including data structures and transformations
3. Detailed Python code with comments showing how to recreate a similar visualization
4. Use matplotlib, seaborn, plotly, or other appropriate visualization libraries
5. Include sample data structures and realistic data generation code

Focus on creating practical, executable code that a researcher could adapt to create similar visualizations with their own data.

Here are the figures from the paper:
"""
        
        # Add information about each figure (no filtering)
        for fig_num, fig_data in viz_figures.items():
            caption = fig_data.get('caption', 'No caption available')
            prompt += f"\n\nFigure {fig_num}: {caption}\n"
            
            # Add context from all references if available
            if fig_data.get('references') and len(fig_data.get('references')) > 0:
                for i, ref in enumerate(fig_data['references'][:3]):  # Include up to 3 references
                    context = ref.get('context', '')
                    prompt += f"Context {i+1}: {context[:300]}...\n"
            
            # Add extended context if available
            if fig_data.get('extended_context') and len(fig_data.get('extended_context')) > 0:
                ext_context = fig_data['extended_context'][0]
                # Extract a shorter version for the prompt
                ext_context_sample = ext_context[:500] + "..." if len(ext_context) > 500 else ext_context
                prompt += f"Extended context: {ext_context_sample}\n"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # Using a compatible model
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert data visualization programmer specialized in scientific figures. Your task is to analyze figure descriptions from a research paper and create detailed Python code instructions that would recreate similar visualizations. Focus on providing practical, executable code with realistic data structures and generation methods. Include thorough comments to explain the visualization approach."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Slightly higher temperature for creative code generation
                max_tokens=4000  # Allow longer response for code
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying OpenAI API: {str(e)}")
            traceback.print_exc()
            return f"Error: {str(e)}"

    def extract_figure_methodology(self):
        """Extract methodology steps for recreating each figure in the paper."""
        # Get figure captions and references
        figures = self.content.get('figures', {})
        
        # Prepare result structure
        figure_methodologies = {}
        
        # Process each figure individually to get focused methodologies
        for fig_num, fig_data in figures.items():
            caption = fig_data.get('caption', 'No caption available')
            
            # Get all context related to this figure
            context_snippets = []
            
            # Add all references if available
            if fig_data.get('references') and len(fig_data.get('references')) > 0:
                for ref in fig_data['references']:
                    context = ref.get('context', '')
                    if context:
                        context_snippets.append(context)
            
            # Add extended context if available
            if fig_data.get('extended_context') and len(fig_data.get('extended_context')) > 0:
                for ext_context in fig_data['extended_context']:
                    if ext_context:
                        context_snippets.append(ext_context)
            
            # Construct a figure-specific prompt
            figure_prompt = f"""
Based on the provided figure caption and surrounding context, describe the methodology steps needed to recreate Figure {fig_num}. 

Focus on:
1. The data processing steps (transformation, normalization, filtering, etc.)
2. Statistical methods or algorithms applied
3. Any parameters or thresholds mentioned
4. Input data requirements and preprocessing

Do NOT include code, only describe the methodology steps in detail.

Figure {fig_num} Caption: {caption}

Context from paper:
"""
            
            # Add context snippets to the prompt
            for i, snippet in enumerate(context_snippets[:5]):  # Limit to 5 snippets to avoid token limits
                figure_prompt += f"\nContext {i+1}: {snippet[:1000]}...\n" if len(snippet) > 1000 else f"\nContext {i+1}: {snippet}\n"
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Using a faster model for individual figure processing
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a scientific methodology expert. Your task is to analyze a figure from a research paper and identify the precise methodology steps needed to recreate it. Focus only on data processing steps and methodologies, not code implementation."
                        },
                        {"role": "user", "content": figure_prompt}
                    ],
                    temperature=0.1,  # Low temperature for factual responses
                    max_tokens=1000  # Shorter response for individual figure
                )
                
                # Store the methodology for this figure
                figure_methodologies[fig_num] = {
                    "caption": caption,
                    "methodology": response.choices[0].message.content
                }
                
            except Exception as e:
                print(f"Error querying OpenAI API for Figure {fig_num}: {str(e)}")
                traceback.print_exc()
                figure_methodologies[fig_num] = {
                    "caption": caption,
                    "methodology": f"Error extracting methodology: {str(e)}"
                }
        
        # Combine all figure methodologies into a formatted string
        result = "# Figure Methodology Steps for Reproduction\n\n"
        
        for fig_num in sorted(figure_methodologies.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            figure_data = figure_methodologies[fig_num]
            result += f"## Figure {fig_num}\n\n"
            result += f"**Caption:** {figure_data['caption']}\n\n"
            result += f"**Methodology Steps:**\n\n{figure_data['methodology']}\n\n"
            result += "---\n\n"
        
        return result

def main():
    try:
        # Content JSON path
        content_json_path = 'extracted_content/all_content.json'
        
        # Check if JSON exists
        if not os.path.exists(content_json_path):
            print(f"Error: Content JSON file not found at {content_json_path}")
            print("Please run extract_all_content.py first to generate the content JSON")
            return

        print("Initializing PaperQueryEngine...")
        query_engine = PaperQueryEngine(content_json_path)
        
        # Query for computational results
        print("\nQuerying for computational results...")
        results = query_engine.query_computational_results()
        
        print("\n=== COMPUTATIONAL RESULTS ===\n")
        print(results)
        
        # Save results to file
        with open('computational_results.txt', 'w') as f:
            f.write(results)
        
        # Query for datasets used
        print("\nQuerying for datasets used...")
        datasets = query_engine.query_datasets()
        
        print("\n=== DATASETS USED ===\n")
        print(datasets)
        
        # Save datasets to file
        with open('datasets_used.txt', 'w') as f:
            f.write(datasets)
            
        # Generate code instructions for figures
        print("\nGenerating code instructions for recreating figures...")
        figure_code = query_engine.generate_figure_code_instructions()
        
        print("\n=== FIGURE CODE INSTRUCTIONS ===\n")
        print(figure_code[:500] + "..." if len(figure_code) > 500 else figure_code)
        
        # Save figure code instructions to file
        with open('figure_code_instructions.txt', 'w') as f:
            f.write(figure_code)
        
        # Extract figure methodology steps
        print("\nExtracting figure methodology steps...")
        figure_methodology = query_engine.extract_figure_methodology()
        
        print("\n=== FIGURE METHODOLOGY STEPS ===\n")
        # Print first 500 characters as preview
        print(figure_methodology[:500] + "..." if len(figure_methodology) > 500 else figure_methodology)
        
        # Save figure methodology to file
        with open('figure_methodology_steps.txt', 'w') as f:
            f.write(figure_methodology)
        
        print("\nResults saved to computational_results.txt, datasets_used.txt, figure_code_instructions.txt, and figure_methodology_steps.txt")
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 