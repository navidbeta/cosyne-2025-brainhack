import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import logging
import base64

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_paper_content():
    """Load the paper content from the JSON file."""
    try:
        with open("extracted_content/all_content.json", "r") as f:
            content = json.load(f)
            if not isinstance(content, dict):
                raise ValueError("Paper content must be a JSON object")
            return content
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading paper content: {str(e)}")
        raise

def analyze_figure_with_context(image_path, paper_content, figure_number):
    """
    Analyze a figure using GPT-4 Vision API along with all relevant text from the paper.
    
    Args:
        image_path (str): Path to the image file
        paper_content (dict): The paper content from JSON
        figure_number (int): The figure number to analyze
        
    Returns:
        str: Comprehensive analysis of the figure and its context
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Get all text content from the paper
        full_text = paper_content.get("full_text", "")
        if not full_text:
            logger.warning("No full text found in paper content")
        
        # Get figure references and their context
        figure_refs = []
        for ref in paper_content.get("figure_references", []):
            if isinstance(ref, dict) and ref.get("figure_number") == figure_number:
                figure_refs.append({
                    "context": ref.get("context", ""),
                    "extended_context": ref.get("extended_context", "")
                })
        
        # Get figure caption
        caption = ""
        for cap in paper_content.get("captions", []):
            if isinstance(cap, dict) and cap.get("figure_number") == figure_number:
                caption = cap.get("text", "")
                break
        
        # Prepare the context for the API
        context_prompt = f"""
        This is Figure {figure_number} from a scientific paper. Here is all the relevant information:

        CAPTION:
        {caption}

        FIGURE REFERENCES IN TEXT:
        {json.dumps(figure_refs, indent=2)}

        FULL PAPER TEXT:
        {full_text}

        Please provide a comprehensive analysis that includes:
        1. Visual description of the figure (type, components, relationships)
        2. Methodology used to create this figure (based on the paper text)
        3. Key findings and conclusions from this figure
        4. How this figure relates to the paper's overall findings
        5. Any subfigures and their specific purposes
        6. The exact methodology and techniques used to generate this figure
        """
        
        # Read the image file and encode it properly
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Call GPT-4 Vision API with both image and text context
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": context_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            # Extract the description
            description = response.choices[0].message.content
            return description
            
    except Exception as e:
        logger.error(f"Error analyzing figure: {str(e)}")
        raise

def main():
    # Path to the figure
    figure_path = "extracted_figures/figures/figure_page8_1.jpeg"
    figure_number = 8  # Extract from filename or specify
    
    try:
        # Load paper content
        paper_content = load_paper_content()
        
        # Analyze the figure with context
        description = analyze_figure_with_context(figure_path, paper_content, figure_number)
        
        # Print the description
        print("\nComprehensive Figure Analysis:")
        print("-" * 50)
        print(description)
        
        # Save the description to a file
        output_dir = "figure_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "figure_page8_1_comprehensive_analysis.txt")
        with open(output_file, "w") as f:
            f.write(description)
            
        print(f"\nAnalysis saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 