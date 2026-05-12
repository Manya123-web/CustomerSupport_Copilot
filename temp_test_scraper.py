# test_ddgs_scraper.py

import sys
import subprocess

# Install ddgs if not already installed
try:
    from ddgs import DDGS
except ImportError:
    print("Installing ddgs library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ddgs"])
    from ddgs import DDGS

def test_web_scraper(query: str, max_results: int = 2):
    print(f"Searching for: {query}\n")
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"URL: {result.get('href', 'N/A')}")
        print(f"Snippet: {result.get('body', 'N/A')[:200]}...")
        print("=" * 80)

if __name__ == "__main__":
    # Change this query to whatever you want
    test_web_scraper("social security retirement benefits")