from langchain_core.tools import tool
import os


from langgraph.prebuilt import ToolNode
import openai

import googlemaps
from dotenv import load_dotenv
load_dotenv(override=True)

API_KEY = os.getenv("GOOGLE_API_KEY")
gmaps = googlemaps.Client(API_KEY)
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
SERP_API_KEY = os.getenv("SERP_API_KEY")

from tools.common_tools import web_search





## Tools for Section 1 - Geographic Location Assessment
@tool
def get_coordinates(poi_name:str) -> dict:
    """Get latitude and longitude of a place name using Google Maps API. This place name can be an asset or a city or a point of interest.

    Args:
        asset_name: Name of the asset (e.g., "Statue of Liberty")
        city: Name of the city (e.g., "New York")
    
    Returns:
        A dictionary containing latitude and longitude of the location.
    """

    search_query = f"{poi_name}"
    geocode_result = gmaps.geocode(search_query)
    if geocode_result:
        location = geocode_result[0]['geometry']['location']
        return {'latitude': location['lat'], 'longitude': location['lng']}
    return None

@tool
def calculate_distance_to_city_centers(asset_coords:dict, city_coords_list:list):
    """Calculate distances from asset coordinates to a list of city center coordinates.
        
    Args:
        asset_coords: A dictionary containing latitude and longitude of the hotel.
        city_coords_list: A list of dictionaries, each containing latitude and longitude of a city center.
    
    """
    distances = []
    for city_coords in city_coords_list:
        distance_result = gmaps.distance_matrix(asset_coords, city_coords)
        if distance_result['rows'][0]['elements'][0]['status'] == 'OK':
            distance = distance_result['rows'][0]['elements'][0]['distance']['value'] / 1000  # Convert to km
            distances.append(distance)
    return distances

@tool
def get_distance_between_coordinates(point1: dict, point2: dict) -> dict:
    """
    Calculates the travel distance in meters between two geographic coordinates using Google Maps Directions API.

    Args:
        point1 (dict): A dictionary with "latitude" and "longitude" for the origin. Example: {"latitude": 43.7, "longitude": 7.3}
        point2 (dict): A dictionary with "latitude" and "longitude" for the destination. Example: {"latitude": 43.8, "longitude": 7.4}

    Returns:
        dict: A dictionary containing:
            - distance_meters (float): Travel distance between the two points in meters.
            - duration_seconds (float): Estimated travel time in seconds.

    Usage Example:
        get_distance_between_coordinates(
            point1={"latitude": 43.7, "longitude": 7.3},
            point2={"latitude": 43.8, "longitude": 7.4}
        )
    """

    origins = (point1["latitude"], point1["longitude"])
    destinations = (point2["latitude"], point2["longitude"])

    result = gmaps.distance_matrix(origins=[origins], destinations=[destinations], mode="driving")

    try:
        element = result["rows"][0]["elements"][0]
        distance_meters = element["distance"]["value"]
        duration_seconds = element["duration"]["value"]
    except (KeyError, IndexError):
        raise RuntimeError("Failed to retrieve distance information from Google Maps.")

    return {
        "distance_meters": distance_meters,
        "duration_seconds": duration_seconds
    }

# Tools for Section 2 - Asset Dimensions
# facciamo solo ricerche online

# Tools for Section 3 - Asset Quality
def analyze_hotel_image_with_azure_openai(image_path_or_url: str) -> dict:
    """
    Sends an image to Azure OpenAI GPT-4o Vision endpoint and asks for an architectural analysis.
    
    Args:
        image_path_or_url (str): Local path or URL to the hotel image.
    
    Returns:
        dict: Analysis including architectural style, main materials, state of conservation, summary.
    """
    # Configura questi valori!
    openai.api_type = "azure"
    openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]  # es: "https://<YOUR-RESOURCE-NAME>.openai.azure.com/"
    openai.api_version = os.environ["OPENAI_API_VERSION"]            # aggiorna se necessario
    openai.api_key = os.environ["AZURE_OPENAI_KEY"]        # la tua chiave Azure OpenAI

    model = os.environ["AZURE_OPENAI_MODEL"]

    prompt = (
        "Analyze this image of a hotel and provide the following:\n"
        "- The architectural style (e.g. modern, classic, Mediterranean, etc.)\n"
        "- Main visible materials (e.g. glass, stone, wood, metal)\n"
        "- General state of conservation (e.g. newly renovated, well-maintained, dated)\n"
        'Return the result as a JSON object with the fields: "architectural_style", "main_materials", "state_of_conservation", "summary".'
    )

    # Gestione dell'immagine locale o remota
    if image_path_or_url.startswith("http"):
        image_input = {"type": "image_url", "image_url": {"url": image_path_or_url}}
    else:
        image_input = {"type": "image_file", "image_file": open(image_path_or_url, "rb")}

    messages = [
        {"role": "user", "content": [prompt, image_input]}
    ]

    # Chiamata all’endpoint vision di Azure OpenAI
    response = openai.chat.completions.create(
        model=model,        # Qui inserisci il nome del deployment Azure
        messages=messages,
        max_tokens=512
    )
    output = response.choices[0].message.content

    # Se vuoi direttamente il JSON, puoi fare:
    try:
        import json
        output_json = json.loads(output)
        return output_json
    except Exception:
        # fallback: ritorna la stringa
        return {"raw_response": output}



# List of tools for initial asset assessment
# These tools will be used by the initial asset assessment agent to gather information about assets and their locations.
initial_asset_analysis_list = [get_coordinates
                                 , calculate_distance_to_city_centers
                                 , get_distance_between_coordinates
                                 , web_search
                                 ]

# Create a ToolNode for the initial asset assessment tools
# This allows the tools to be used in a structured way within the LangGraph framework.
initial_asset_analysis_tools = ToolNode(initial_asset_analysis_list)