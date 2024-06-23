import requests
from bs4 import BeautifulSoup

def fetch_arxiv_papers(query, max_results=10):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'xml')
        papers = soup.find_all('entry')
        paper_list = []
        for paper in papers:
            title = paper.title.text.strip()
            summary = paper.summary.text.strip()
            paper_list.append({"title": title, "summary": summary})
        return paper_list
    except requests.RequestException as e:
        print(f"Error fetching arXiv papers: {e}")
        return []