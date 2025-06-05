from langchain_core.tools import tool
import os
import requests
from PIL import Image
from io import BytesIO
import math
import numpy as np
from langchain.agents import Tool
from langgraph.prebuilt import ToolNode
import base64
from segment_anything import sam_model_registry, SamPredictor

import googlemaps
API_KEY = os.getenv("GOOGLE_API_KEY")
gmaps = googlemaps.Client(key=API_KEY)
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")


from dotenv import load_dotenv
load_dotenv(override=True)



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
    Calculates the travel distance in meters between two geographic coordinates using Google Maps Distance Matrix API.

    Args:
        point1 (dict): A dictionary with "latitude" and "longitude" for the origin.
        point2 (dict): A dictionary with "latitude" and "longitude" for the destination.

    Returns:
        dict: A dictionary containing:
            - distance_meters (float): Travel distance between the two points in meters.
            - duration_seconds (float): Estimated travel time in seconds.
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

@tool
def download_satellite_image(latitude: float, longitude: float, zoom: int = 18, image_size: int = 512) -> Image.Image:
    """
    Downloads a satellite image centered at the given coordinates.

    Args:
        latitude (float): The latitude of the center point.
        longitude (float): The longitude of the center point.
        zoom (int): Zoom level for the satellite image (default: 18).
        image_size (int): Size of the image in pixels (default: 512x512).

    Returns:
        PIL.Image.Image: A satellite image centered at the specified location.
    """
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{longitude},{latitude},{zoom}/{image_size}x{image_size}?access_token={MAPBOX_TOKEN}"
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download image: {response.status_code} - {response.text}")
    return Image.open(BytesIO(response.content))

@tool
def estimate_scale(latitude: float, zoom: int) -> dict:
    """
    Estimates the map scale in meters per pixel at a given latitude and zoom level.

    Args:
        latitude (float): The latitude of the location.
        zoom (int): The map zoom level.

    Returns:
        dict: A dictionary containing:
            - meters_per_pixel (float): The estimated number of meters per pixel.
    """
    tile_size = 256
    earth_radius = 6378137  # meters
    meters_per_pixel = (
        math.cos(math.radians(latitude)) *
        2 * math.pi * earth_radius /
        (tile_size * 2**zoom)
    )
    return {
        "meters_per_pixel": meters_per_pixel
    }

@tool
def calculate_area(binary_mask: list[list[int]], meters_per_pixel: float) -> dict:
    """
    Calculates the area in square meters of a building using a binary mask and scale.

    Args:
        binary_mask (list[list[int]]): A 2D binary mask where 1 indicates the building area.
        meters_per_pixel (float): The scale of the image in meters per pixel.

    Returns:
        dict: A dictionary containing:
            - area_m2 (float): The estimated area of the building in square meters.
    """
    mask_array = np.array(binary_mask)
    pixel_count = np.sum(mask_array == 1)
    area_m2 = pixel_count * (meters_per_pixel ** 2)
    return {
        "area_m2": area_m2
    }

def calculate_scale(point1: dict, point2: dict, distance_meters: float) -> dict:
    """
    Calculates the scale of a satellite image in meters per pixel, based on the real-world distance
    between two points and their distance in pixels in the image.

    Args:
        point1 (dict): Dictionary with keys "x" and "y" representing the first point in pixel coordinates.
        point2 (dict): Dictionary with keys "x" and "y" representing the second point in pixel coordinates.
        distance_meters (float): Real-world distance between the two points in meters.

    Returns:
        dict: A dictionary containing:
            - meters_per_pixel (float): The scale in meters per pixel.
    """
    dx = point2["x"] - point1["x"]
    dy = point2["y"] - point1["y"]
    pixel_distance = (dx ** 2 + dy ** 2) ** 0.5
    meters_per_pixel = distance_meters / pixel_distance

    return {
        "meters_per_pixel": meters_per_pixel
    }

@tool
def segment_building_base64(image_base64: str, predictor: SamPredictor) -> list[list[int]]:
    """
    Segments a building in a satellite image using SAM from a base64-encoded image.

    Args:
        image_base64 (str): Base64-encoded PNG or JPEG image.
        predictor (SamPredictor): Initialized SAM predictor.

    Returns:
        list[list[int]]: A binary mask as nested lists where 1 indicates the segmented building.
    """
    # Decode base64 to image
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)

    predictor.set_image(image_np)

    h, w, _ = image_np.shape
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]
    Image.fromarray((best_mask.astype(np.uint8) * 255).astype(np.uint8)).save("mask.png")
    return best_mask.astype(int).tolist()


initial_asset_assessment_list = [get_coordinates,
                                 calculate_distance_to_city_centers,
                                 get_distance_between_coordinates,
                                 download_satellite_image,
                                 estimate_scale,
                                 calculate_area,
                                 calculate_scale,
                                 segment_building_base64]

initial_asset_assessment_tools = ToolNode(initial_asset_assessment_list)