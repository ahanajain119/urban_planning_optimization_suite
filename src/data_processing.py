import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import folium
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from shapely.geometry import box
import shapely.ops
import pyproj
import streamlit as st
from folium.plugins import MiniMap, Fullscreen, MarkerCluster
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Census data for cities
CENSUS_POPULATION = {
    "mysore": 920550  
}

# Define GREEN_TAGS locally for use in this module only
GREEN_TAGS = [
    'forest', 'grass', 'recreation_ground', 'park', 'playground',
    'sports_centre', 'stadium', 'wood', 'grassland'
]

# Data file path constants
ROAD_NETWORK_PATH = 'data/raw/mysore_road_network.graphml'
BUILDINGS_PATH = 'data/raw/mysore_buildings.geojson'
AMENITIES_PATH = 'data/raw/mysore_amenities.geojson'
LANDUSE_PATH = 'data/raw/mysore_landuse.geojson'
POIS_PATH = 'data/raw/mysore_pois.geojson'

# BBox offset constant
BBOX_OFFSET = 0.03

# Key areas in Mysore for analysis
class MysoreDataLoader:
    """
    MysoreDataLoader is a specialized class for loading, processing, and analyzing urban spatial data for the city of Mysore.
    It provides methods to fetch data from OpenStreetMap, process key features (roads, buildings, amenities, land use, POIs),
    and compute basic city statistics for urban planning and optimization tasks.
    """
    
    def __init__(self):
        """Initialize Mysore data loader with city-specific parameters"""
        # Mysore city center coordinates (around Mysore Palace)
        self.city_center = (12.3051, 76.6553)
        
        # Define analysis area (5km x 5km around city center)
        self.bbox = {
            'north': self.city_center[0] + BBOX_OFFSET,   # +0.03 degrees (~3.3 km north)
            'south': self.city_center[0] - BBOX_OFFSET,   # -0.03 degrees (~3.3 km south)
            'east': self.city_center[1] + BBOX_OFFSET,    # +0.03 degrees (~3.3 km east)
            'west': self.city_center[1] - BBOX_OFFSET     # -0.03 degrees (~3.3 km west)
        }
        
        # Key areas in Mysore for analysis
        self.key_locations = {
            'Mysore Palace': (12.3051, 76.6553),
            'Devaraja Market': (12.3069, 76.6553),
            'University of Mysore': (12.3170, 76.6480),
            'Mysore Railway Station': (12.3079, 76.6466),
            'Chamundi Hills': (12.2785, 76.6725),
            'Kukkarahalli Lake': (12.3276, 76.6424),
            'Karanji Lake': (12.3107, 76.6437),
            'JSS Medical College': (12.3367, 76.6121)
        }
        
        # Major roads to analyze
        self.major_roads = [
            'Sayyaji Rao Road',
            'JLB Road', 
            'Hunsur Road',
            'Bannur Road',
            'Ooty Road',
            'Vinoba Road'
        ]
        
        # Data storage
        self.road_network = None
        self.buildings = None
        self.amenities = None
        self.land_use = None
        self.pois = None  # Points of Interest

    def load_mysore_data(self, save_to_file: bool = True) -> Dict:
        """
        Load complete Mysore city data from OpenStreetMap or local cache if available.
        Args:
            save_to_file: Whether to save loaded data to files
        Returns:
            Dictionary containing all loaded data
        """
        logger.info("Loading Mysore city data...")

        # Use constants for file paths
        road_network_path = ROAD_NETWORK_PATH
        buildings_path = BUILDINGS_PATH
        amenities_path = AMENITIES_PATH
        landuse_path = LANDUSE_PATH
        pois_path = POIS_PATH

        try:
            # Load from disk if files exist
            files_exist = all([
                os.path.exists(road_network_path),
                os.path.exists(buildings_path),
                os.path.exists(amenities_path),
                os.path.exists(landuse_path),
                os.path.exists(pois_path)
            ])
            if files_exist:
                logger.info("Loading data from local cache...")
                self.road_network = ox.load_graphml(road_network_path)
                self.buildings = gpd.read_file(buildings_path)
                self.amenities = gpd.read_file(amenities_path)
                self.land_use = gpd.read_file(landuse_path)
                self.pois = gpd.read_file(pois_path)
            else:
                # Download from OSM 
                logger.info("Downloading data from OpenStreetMap...")
                # Load road network
                logger.info("Loading road network...")
                self.road_network = ox.graph_from_bbox(
                    self.bbox['north'],
                    self.bbox['south'],
                    self.bbox['east'],
                    self.bbox['west'],
                    network_type='drive',
                    simplify=True,
                    retain_all=False
                )
                # Convert to GeoDataFrame for easier analysis
                nodes, edges = ox.graph_to_gdfs(self.road_network)
                # Load buildings
                logger.info("Loading buildings...")
                self.buildings = ox.features_from_bbox(
                    self.bbox['north'], 
                    self.bbox['south'],
                    self.bbox['east'], 
                    self.bbox['west'],
                    tags={'building': True}
                )
                # Load amenities
                logger.info("Loading amenities...")
                amenity_tags = {
                    'amenity': ['school', 'hospital', 'clinic', 'bank', 'restaurant', 
                            'cafe', 'pharmacy', 'fuel', 'police', 'fire_station',
                            'library', 'university', 'college', 'marketplace']
                }
                self.amenities = ox.features_from_bbox(
                    self.bbox['north'], 
                    self.bbox['south'], 
                    self.bbox['east'], 
                    self.bbox['west'],
                    tags=amenity_tags
                )
                # Load land use
                logger.info("Loading land use...")
                landuse_tags = {
                    'landuse': ['residential', 'commercial', 'industrial', 
                            'forest', 'grass', 'recreation_ground'],
                    'leisure': ['park', 'playground', 'sports_centre', 'stadium'],
                    'natural': ['water', 'wood', 'grassland']
                }
                self.land_use = ox.features_from_bbox(
                    self.bbox['north'], 
                    self.bbox['south'], 
                    self.bbox['east'], 
                    self.bbox['west'],
                    tags=landuse_tags
                )
                # Load Points of Interest
                logger.info("Loading points of interest...")
                poi_tags = {
                    'tourism': ['attraction', 'museum', 'monument'],
                    'historic': ['palace', 'monument', 'building'],
                    'shop': ['mall', 'supermarket', 'market']
                }
                self.pois = ox.features_from_bbox(
                    self.bbox['north'], 
                    self.bbox['south'], 
                    self.bbox['east'], 
                    self.bbox['west'],
                    tags=poi_tags
                )
                logger.info("Mysore data loaded successfully!")
                # Save to files if requested
                if save_to_file:
                    self._save_data_to_files()

            return {
                'road_network': self.road_network,
                'buildings': self.buildings,
                'amenities': self.amenities,
                'land_use': self.land_use,
                'pois': self.pois,
                'bbox': self.bbox,
                'key_locations': self.key_locations
            }
        except Exception as e:
            logger.error(f"❌ Error loading Mysore data: {e}")
            raise
    
    def _save_data_to_files(self):
        """Save loaded data to files for future use"""
        try:
            # Save road network
            ox.save_graphml(self.road_network, ROAD_NETWORK_PATH)
            
            # Save GeoDataFrames
            if self.buildings is not None:
                self.buildings.to_file(BUILDINGS_PATH, driver='GeoJSON')
            
            if self.amenities is not None:
                self.amenities.to_file(AMENITIES_PATH, driver='GeoJSON')
            
            if self.land_use is not None:
                self.land_use.to_file(LANDUSE_PATH, driver='GeoJSON')
            
            if self.pois is not None:
                self.pois.to_file(POIS_PATH, driver='GeoJSON')
            
            logger.info("💾 Data saved to files successfully!")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def get_basic_stats(self) -> Dict:
        """
        Calculate basic statistics about Mysore
        
        Returns:
            Dictionary with basic city statistics
        """
        stats = {}
        
        try:
            # Road network stats
            if self.road_network:
                stats['total_roads'] = len(self.road_network.edges())
                stats['total_intersections'] = len(self.road_network.nodes())
                stats['road_length_km'] = sum(
                    [data.get('length', 0) for _, _, data in self.road_network.edges(data=True)]
                ) / 1000
            
            # Building stats
            if self.buildings is not None:
                stats['total_buildings'] = len(self.buildings)
                
            # Amenity stats
            if self.amenities is not None:
                stats['total_amenities'] = len(self.amenities)
                amenity_types = self.amenities['amenity'].value_counts().to_dict()
                stats['amenity_breakdown'] = amenity_types
            
            # Land use stats
            if self.land_use is not None:
                stats['total_land_parcels'] = len(self.land_use)
                if 'landuse' in self.land_use.columns:
                    landuse_types = self.land_use['landuse'].value_counts().to_dict()
                    stats['landuse_breakdown'] = landuse_types
            
            # Analysis area
            stats['analysis_area_km2'] = 25  # 5km x 5km
            stats['city_center'] = self.city_center
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error calculating stats: {e}")
            return stats
    
    def preview_data(self):
        """
        Print a preview of loaded data
        """
        print("🏛️ MYSORE CITY DATA PREVIEW")
        print("=" * 50)
        
        if self.road_network:
            print(f"🛣️  Roads: {len(self.road_network.edges())} segments")
            print(f"🔄 Intersections: {len(self.road_network.nodes())} nodes")
        
        if self.buildings is not None:
            print(f"🏢 Buildings: {len(self.buildings)} structures")
        
        if self.amenities is not None:
            print(f"🏥 Amenities: {len(self.amenities)} facilities")
            print("   Top amenities:")
            if 'amenity' in self.amenities.columns:
                for amenity, count in self.amenities['amenity'].value_counts().head(5).items():
                    print(f"     • {amenity}: {count}")
        
        if self.land_use is not None:
            print(f"🌳 Land parcels: {len(self.land_use)} areas")
        
        print(f"📍 Analysis area: {self.bbox}")
        print(f"🎯 Key locations: {len(self.key_locations)} landmarks")

# Test the data loader
if __name__ == "__main__":
    # Create data loader
    loader = MysoreDataLoader()
    
    # Load data
    data = loader.load_mysore_data(save_to_file=False)
    
    if data:
        # Show preview
        loader.preview_data()
        
        # Get basic stats
        stats = loader.get_basic_stats()
        print("\n📊 BASIC STATISTICS:")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  • {k}: {v}")
            else:
                print(f"{key}: {value}")
    
    print("\n🎯 Next step: Run this to load Mysore data!")

def calculate_area_km2(bbox):
    geom = box(bbox['west'], bbox['south'], bbox['east'], bbox['north'])
    # Project to UTM zone 43N for Mysore (EPSG:32643)
    proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    projected = shapely.ops.transform(proj.transform, geom)
    return projected.area / 1e6  # m² to km²

#Green space percentage calculation
def calculate_green_space_pct(land_use_gdf, total_area_km2):
    if land_use_gdf is None or land_use_gdf.empty:
        return 0
    
    # Check for relevant columns and filter
    mask = (
        land_use_gdf.get('landuse', pd.Series()).isin(GREEN_TAGS) |
        land_use_gdf.get('leisure', pd.Series()).isin(GREEN_TAGS) |
        land_use_gdf.get('natural', pd.Series()).isin(GREEN_TAGS)
    )
    green_gdf = land_use_gdf[mask]
    if green_gdf.empty:
        return 0
    
    # Cache projected GeoDataFrame to avoid repeated CRS transformations
    if not hasattr(green_gdf, '_proj_cache'):
        green_gdf._proj_cache = green_gdf.to_crs(epsg=32643)
    green_gdf_proj = green_gdf._proj_cache
    green_area = green_gdf_proj.geometry.area.sum() / 1e6  # m² to km²
    return round(100 * green_area / total_area_km2, 2) if total_area_km2 > 0 else 0


# Traffic score proxy using average node degree ---
def calculate_traffic_score(road_network):
    if road_network is None:
        return 0
    degrees = dict(road_network.degree())
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    # Normalize to a 1-10 scale
    return round(min(10, max(1, avg_degree * 2)), 2)

@st.cache_data(show_spinner="Loading Mysore city data...")
def get_city_stats(city: str):
    city_key = city.lower()
    if city_key == "mysore":
        loader = MysoreDataLoader()
        loader.load_mysore_data(save_to_file=False)
        stats = loader.get_basic_stats()
        # Population from census
        population = CENSUS_POPULATION.get(city_key, 0)
        # Area from OSM bbox
        area = calculate_area_km2(loader.bbox)
        # Green space %
        green_space_pct = calculate_green_space_pct(loader.land_use, area)
        # Traffic score proxy
        traffic_score = calculate_traffic_score(loader.road_network)
        land_use = stats.get("landuse_breakdown", None)
        return {
            "population": population,
            "area": round(area, 2),
            "green_space_pct": green_space_pct,
            "traffic_score": traffic_score,
            "land_use": land_use   
        }
    # If city not recognized, return empty stats
    st.warning(f"City '{city}' not recognized. Please choose a supported city.")
    return {
        "population": 0,
        "area": 0,
        "green_space_pct": 0,
        "traffic_score": 0,
        "land_use": None
    }
