import gradio as gr
import numpy as np
from cohere_integration import summarize_text
from data_collection import fetch_arxiv_papers
from preprocessing import preprocess_documents
from topic_modeling import perform_lda, visualize_lda
import matplotlib.pyplot as plt

def research_assistant(query, max_results):
    # Fetch arXiv papers
    papers = fetch_arxiv_papers(query, max_results)
    if not papers:
        return "No papers found or error in fetching papers.", plt.figure()

    # Preprocess and summarize
    summaries = [summarize_text(paper['summary']) for paper in papers]
    preprocessed_summaries = preprocess_documents(summaries)

    # Check if we have any valid preprocessed summaries
    if not any(preprocessed_summaries):
        return "Error in preprocessing documents.", plt.figure()

    # Perform topic modeling
    vocab = list(set.union(*map(set, preprocessed_summaries)))
    X = np.array([np.bincount([vocab.index(word) for word in doc if word in vocab], minlength=len(vocab)) for doc in preprocessed_summaries])
    y = np.arange(len(papers))  # Assign a unique label to each paper

    # Ensure we have at least two components for visualization
    num_components = min(2, len(papers))
    
    try:
        X_lda, W = perform_lda(X, y, num_components=num_components)
    except Exception as e:
        print(f"Error in topic modeling: {e}")
        return "Error in performing topic modeling.", plt.figure()

    # Visualize results
    titles = [paper['title'] for paper in papers]
    fig = visualize_lda(X_lda, titles)

    # Prepare output
    results = []
    for i, paper in enumerate(papers):
        results.append(f"Title: {paper['title']}\nSummary: {summaries[i]}\n")

    return "\n".join(results), fig

# Create Gradio interface
iface = gr.Interface(
    fn=research_assistant,
    inputs=[
        gr.Textbox(label="Search Query"),
        gr.Slider(minimum=1, maximum=50, step=1, label="Max Results", value=10)
    ],
    outputs=[
        gr.Textbox(label="Research Results"),
        gr.Plot(label="Topic Modeling Visualization")
    ],
    title="Research Assistant Agent",
    description="Enter a search query to fetch and analyze arXiv papers."
)

# Launch the interface
iface.launch()