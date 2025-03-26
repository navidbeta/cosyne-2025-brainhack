# Scientific Paper Methodology Extractor

This tool analyzes scientific papers to extract detailed computational methodology, enabling reproduction of the results when you already have access to the required datasets.

## Overview

The Scientific Paper Methodology Extractor processes PDF papers through several stages:

1. **Text Extraction**: Extracts all text content from the PDF
2. **Metadata Extraction**: Identifies paper title, authors, publication details, and abstract
3. **Computational Finding Identification**: Locates the key computational results in the paper
4. **Methodology Analysis**: Provides detailed step-by-step instructions to reproduce each computational finding

## Key Features

- Processes any scientific PDF paper
- Focuses specifically on computational methods and algorithms
- Extracts implementation details and exact parameters
- Provides step-by-step reproduction instructions
- Identifies potential issues and troubleshooting steps

## Requirements

- Python 3.7+
- OpenAI API key (set in `.env` file)
- Required Python packages (install via `pip install -r requirements.txt`):
  - PyMuPDF
  - openai
  - python-dotenv
  - tqdm

## Installation

1. Clone this repository
2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script with a path to your scientific paper PDF:

```bash
python analyze_paper.py path/to/paper.pdf
```

By default, the script saves analysis results to the `paper_analysis` directory. You can specify a custom output directory using the `-o` or `--output` parameter:

```bash
python analyze_paper.py path/to/paper.pdf -o custom_output_dir
```

The script will process the paper and create a structured output in the specified directory:

```
output_directory/
├── full_text.txt                       # Complete text content of the paper
├── metadata.json                       # Paper metadata (title, authors, etc.)
├── findings.json                       # All computational findings
├── summary_report.json                 # Analysis summary
└── computational_findings/             # Detailed methodology for each finding
    ├── finding_1_analysis.txt
    ├── finding_2_analysis.txt
    └── ...
```

## Understanding the Output

Each computational finding analysis includes:

1. **Identification**: Where the finding appears in the paper
2. **Methodology Extraction**: Detailed description of methods, tools, and parameters
3. **Implementation Details**: Step-by-step implementation instructions
4. **Reproduction Procedure**: Exact steps to reproduce with your existing data
5. **Expected Results**: What output to expect and how to validate it
6. **Troubleshooting**: Potential issues and how to address them

## Example

For a computational finding like "A novel deep learning model achieved 95% accuracy on the task":

- The analysis would include the exact architecture details
- All hyperparameters and training procedures
- Data preprocessing steps
- Evaluation methodology
- Implementation details sufficient to reproduce the 95% accuracy

## Limitations

- The quality of extraction depends on how well the methods are documented in the paper
- Some papers may not include sufficient implementation details for perfect reproduction
- The tool works best on papers with clear methodology sections

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# Paper Analysis Toolkit

This toolkit provides utilities for extracting content from scientific papers and analyzing them to:
- Extract all text content and figure captions
- Identify the 5 most important computational results and methodologies
- Determine what datasets were used in the paper
- Generate code instructions for recreating similar figures

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Configure your OpenAI API key:
   - Add your OpenAI API key to the `.env` file
   - Format: `OPENAI_API_KEY=your_api_key_here`

## Usage

### 1. Extract Content from PDF

First, extract all content from your PDF file:

```bash
python extract_all_content.py
```

This will create a JSON file with all extracted content at `extracted_content/all_content.json`.

### 2. Analyze the Paper Content

Once the content is extracted, run the analysis script:

```bash
python paper_query.py
```

This will generate three output files:
- `computational_results.txt`: The 5 most important computational results and methodologies
- `datasets_used.txt`: Analysis of datasets used in the paper
- `figure_code_instructions.txt`: Code instructions for recreating similar figures

## Customization

You can modify the scripts to:
- Change the PDF file path in `extract_all_content.py`
- Adjust the OpenAI models used in `paper_query.py`
- Add new types of analysis by extending the `PaperQueryEngine` class

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages listed in `requirements.txt`

# Paper Figure Extractor

This script downloads open source papers and extracts all figures and their captions. It's particularly useful for researchers who want to analyze or reproduce figures from scientific papers.

## Features

- Downloads PDF papers from URLs
- Extracts all figures from the PDF
- Extracts figure captions
- Saves figures in their original format
- Creates a detailed JSON summary of all extracted content
- Shows progress bars for downloads and processing
- Handles various PDF formats and image types

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with a paper URL as argument:

```bash
python download_paper_figures.py <paper_url>
```

For example:
```bash
python download_paper_figures.py https://example.com/paper.pdf
```

## Output

The script creates an `extracted_figures` directory containing:
- `paper.pdf`: The downloaded paper
- `figures/`: Directory containing all extracted figures
- `summary.json`: JSON file with metadata about the extracted content

The summary.json file includes:
- Paper URL
- PDF file path
- List of all figures with their:
  - Page number
  - Index on page
  - Filename
  - Path
  - Dimensions
  - Format
- List of all figure captions with their:
  - Page number
  - Caption text
- Total counts of figures and captions

## Notes

- The script works best with open source papers that are directly accessible via URL
- Some papers may require authentication or have access restrictions
- Figure quality depends on the PDF quality and how the figures were embedded
- The script uses PyMuPDF (fitz) for PDF processing, which is more reliable for figure extraction than other PDF libraries 