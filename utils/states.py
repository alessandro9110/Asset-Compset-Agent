from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Annotated, Dict, List, Optional
from langgraph.graph.message import add_messages, AnyMessage

class CityInfo(BaseModel):
    name: str = Field(description="Name of the city")
    coordinates: Dict[str, float] = Field(description="Coordinates of the city in latitude and longitude")
    distance_km: float  = Field(description="Distance from the hotel to the city center in kilometers")

class PositionAnalysis(BaseModel):
    hotel_coordinates: Dict[str, float] = Field(description="Coordinates of the hotel in latitude and longitude")
    nearby_cities: List[CityInfo]  = Field(description="List of nearby cities with their coordinates and distances from the hotel")
    context: str =  Field(description="Contextual information about the hotel and its surroundings (e.g. urban, semi-urban, isolated)")     
    accessibility: str = Field(description="Accessibility of the hotel (e.g. well-connected, remote, etc.)")
    summary: str = Field(description="Summary of the position analysis")
class ValueWithSource(BaseModel):
    value: Optional[str] = Field(description="Extracted value, or null if not found")
    source_url: Optional[str] = Field(description="Source URL from which the value was extracted, or null")

class AmenityWithSource(BaseModel):
    amenity: str = Field(description="Name of the amenity")
    source_url: Optional[str] = Field(description="Source URL for the amenity, if available")

class AssetDimensions(BaseModel):
    number_of_rooms: Optional[ValueWithSource] = Field(description="Number of rooms with source")
    total_surface_area_sqm: Optional[ValueWithSource] = Field(description="Total surface area in square meters with source")
    main_amenities: Optional[List[AmenityWithSource]] = Field(description="Main amenities and their sources")
    additional_information: Optional[str] = Field(description="Any extra info relevant to asset size/potential")


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[list[AnyMessage], add_messages]
    position_analysis: PositionAnalysis
    asset_dimensions: AssetDimensions 