# Research Assistant Agent

A tool that fetches, summarizes, and analyzes arXiv papers using natural language processing and topic modeling.

## Features

- Fetch papers from arXiv based on user queries
- Summarize paper abstracts using Cohere
- Preprocess text data for analysis
- Perform topic modeling using LDA (with SVD fallback)
- Visualize paper relationships in a 2D space
- User-friendly interface powered by Gradio

## Technical Stack

- Python 3.11+
- Libraries:
  - Cohere
  - Requests
  - BeautifulSoup4
  - NLTK
  - NumPy
  - Matplotlib
  - Scikit-learn
  - Gradio

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up a `.env` file with your Cohere API key
4. Run the main script: `python gradio_interface.py`

## Usage

Enter a search query and specify the number of papers to retrieve. The tool will fetch papers from arXiv, summarize them, perform topic modeling, and display the results along with a visualization.

## License

[MIT License](LICENSE)
