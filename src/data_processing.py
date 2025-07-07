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

#Census data for cities
CENSUS_POPULATION = {
    "mysore": 920550  # Replace with actual census value if you have a more accurate one
}

class MysoreDataLoader:
            """Specialized data loader for Mysore city analysis"""
        
            def __init__(self):
                """Initialize Mysore data loader with city-specific parameters"""
                # Mysore city center coordinates (around Mysore Palace)
                self.city_center = (12.3051, 76.6553)
                
                # Define analysis area (5km x 5km around city center)
                self.bbox = {
                    'north': 12.3351,   # +0.03 degrees (~3.3 km north)
                    'south': 12.2751,   # -0.03 degrees (~3.3 km south)
                    'east': 76.6853,    # +0.03 degrees (~3.3 km east)
                    'west': 76.6253     # -0.03 degrees (~3.3 km west)
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
                Load complete Mysore city data from OpenStreetMap
                
                Args:
                    save_to_file: Whether to save loaded data to files
                    
                Returns:
                    Dictionary containing all loaded data
                """
                print("Loading Mysore city data...")
                
                try:
                    # Load road network
                    print("Loading road network...")
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
                    print("Loading buildings...")
                    self.buildings = ox.features_from_bbox(
                        self.bbox['north'], 
                        self.bbox['south'],
                        self.bbox['east'], 
                        self.bbox['west'],
                        tags={'building': True}
                    )
                    
                    # Load amenities
                    print("Loading amenities...")
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
                    print("Loading land use...")
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
                    print("Loading points of interest...")
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
                    
                    print("Mysore data loaded successfully!")
                    
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
                    print(f"❌ Error loading Mysore data: {e}")
                    return None
            
            def _save_data_to_files(self):
                """Save loaded data to files for future use"""
                try:
                    # Save road network
                    ox.save_graphml(self.road_network, 'data/raw/mysore_road_network.graphml')
                    
                    # Save GeoDataFrames
                    if self.buildings is not None:
                        self.buildings.to_file('data/raw/mysore_buildings.geojson', driver='GeoJSON')
                    
                    if self.amenities is not None:
                        self.amenities.to_file('data/raw/mysore_amenities.geojson', driver='GeoJSON')
                    
                    if self.land_use is not None:
                        self.land_use.to_file('data/raw/mysore_landuse.geojson', driver='GeoJSON')
                    
                    if self.pois is not None:
                        self.pois.to_file('data/raw/mysore_pois.geojson', driver='GeoJSON')
                    
                    print("💾 Data saved to files successfully!")
                    
                except Exception as e:
                    print(f"Error saving data: {e}")
            
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
                    print(f"❌ Error calculating stats: {e}")
                    return stats
            
            def create_base_map(self) -> folium.Map:
                """
                Create a base map of Mysore for visualization
                
                Returns:
                    Folium map object
                """
                # Create base map centered on Mysore Palace
                m = folium.Map(
                    location=self.city_center,
                    zoom_start=13,
                    tiles='OpenStreetMap'
                )
                
                # Add key locations
                for location, coords in self.key_locations.items():
                    folium.Marker(
                        location=coords,
                        popup=f"📍 {location}",
                        tooltip=location,
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(m)
                
                # Add analysis boundary
                folium.Rectangle(
                    bounds=[[self.bbox['south'], self.bbox['west']], 
                        [self.bbox['north'], self.bbox['east']]],
                    color='blue',
                    weight=2,
                    fill=False,
                    popup='Analysis Area (5km x 5km)'
                ).add_to(m)
                
                return m
            
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
    green_tags = [
        'forest', 'grass', 'recreation_ground', 'park', 'playground',
        'sports_centre', 'stadium', 'wood', 'grassland'
    ]
    # Check for relevant columns and filter
    mask = (
        land_use_gdf.get('landuse', pd.Series()).isin(green_tags) |
        land_use_gdf.get('leisure', pd.Series()).isin(green_tags) |
        land_use_gdf.get('natural', pd.Series()).isin(green_tags)
    )
    green_gdf = land_use_gdf[mask]
    if green_gdf.empty:
        return 0
    #accurate area calculation
    green_gdf_proj = green_gdf.to_crs(epsg=32643)
    green_area = green_gdf_proj.geometry.area.sum() / 1e6  # m² to km²
    return round(100 * green_area / total_area_km2, 2) if total_area_km2 > 0 else 0

# Traffic score proxy using average node degree ---
def calculate_traffic_score(road_network):
    if road_network is None:
        return 0
    degrees = dict(road_network.degree())
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    # Normalize to a 1-10 scale (tune as needed)
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
            "traffic_score": 0
        }