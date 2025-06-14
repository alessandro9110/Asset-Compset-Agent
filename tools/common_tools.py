import requests
import os

from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv(override=True)


SERP_API_KEY = os.getenv("SERP_API_KEY")


@tool
def web_search(query: str) -> str:
    """
    Performs a Google web search using SerpAPI and returns the main text snippets from the top results and the urls

    Args:
        query (str): The search query (e.g., 'Les Terrasses d’Eze number of rooms surface spa').

    Returns:
        list: List of dicts with 'snippet' and 'link' for each result

    Example:
        >>> web_search("Les Terrasses d’Eze number of rooms surface spa")
        'Les Terrasses d’Eze offers 87 rooms and suites...'

    Note:
        You need to set your SerpAPI key in the SERPAPI_KEY variable.
        Get a free key at https://serpapi.com/.
    """
    SERPAPI_KEY = SERP_API_KEY 
    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "engine": "google",
        "api_key": SERPAPI_KEY,
        "num": 10,
        "hl": "it"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    results = []
    for res in data.get("organic_results", []):
        if "snippet" in res:
            results.append({
                "snippet": res["snippet"],
                "link": res.get("link")
            })
    return results

