# Import the DuckDuckGo search
from langchain_community.tools import DuckDuckGoSearchRun

# Import the generic tool wrapper from LangChain
from langchain_core.tools import tool

# Standard libraries for scraping and saving
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re


search = DuckDuckGoSearchRun()

# Search for the query propmted by the user on the web 
@tool("search_tool")
def search_website(query: str) -> str:
    """Search DuckDuckGo to discover an initial list of local companies in a specific city"""
    try:
        return search.invoke({"query": query})
        
    except Exception as e:
        return f"Error searching on DuckDuckGo: {e}"

# Scrape raw text from a website
def scrape_website(url: str) -> str:
    try:
        # Send GET request to the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad HTTP codes

        # Parse and clean up the raw HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        # Limit to 3000 characters for performance and API limits
        return text[:3000]
    except Exception as e:
        return f"Error scraping website: {e}"

# Generate search queries to look for IT services related to a company
def generate_search_queries(company_name: str) -> list[str]:
    keywords = ["IT Services", "managed IT", "technology solutions"]
    return [f"{company_name} {keyword}" for keyword in keywords]

# Combined search and scrape operation for a company
@tool("scrape_tool")
def search_and_scrape(query: str) -> str:
    """Search the web for information. Scrape the content of the website and search for related information"""
    queries = generate_search_queries(query)
    results = []

    for i in queries:
        # Run web search
        search_results = search.invoke({"query": i})

        # Extract URLs from the search output
        urls = re.findall(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            search_results
        )

        # Scrape the first valid URL found
        if urls:
            results.append(scrape_website(urls[0]))

    # Combine all the text results into one big chunk
    return " ".join(results)

# Save-to-text tool: saves the output to a text file
@tool("save_tool")
def save_to_txt(data: str, filename: str = "leads_output.txt"):
    """Saves structured data to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Leads Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    # Open the file in append mode so it keeps growing over time
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"