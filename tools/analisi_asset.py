from langchain_core.tools import tool
import os

import googlemaps
googlemaps_api_key = os.getenv("GOOGLE_API_KEY")
gmaps = googlemaps.Client(key=googlemaps_api_key)


from dotenv import load_dotenv
load_dotenv(override=True)

# EXAMPLE TOOL
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# EXAMPLE TOOL
@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

# EXAMPLE TOOL
@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


@tool
def get_asset_coordinates(asset_name:str, city:str) -> dict:
    """Get latitude and longitude of an asset using Google Maps API.

    Args:
        asset_name: Name of the asset (e.g., "Statue of Liberty")
        city: Name of the city (e.g., "New York")
    
    Returns:
        A dictionary containing latitude and longitude of the location.
    """

    search_query = f"{asset_name}, {city}"
    geocode_result = gmaps.geocode(search_query)
    if geocode_result:
        location = geocode_result[0]['geometry']['location']
        return {'latitude': location['lat'], 'longitude': location['lng']}
    return None

@tool
def get_city_coordinates(city:str) -> dict:
    """Get latitude and longitude of a city set using Google Maps API.

    Args:
        asset_name: Name of the asset (e.g., "Statue of Liberty")
        city: Name of the city (e.g., "New York")
    
    Returns:
        A dictionary containing latitude and longitude of the location.
    """

    search_query = f"{city}"
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
