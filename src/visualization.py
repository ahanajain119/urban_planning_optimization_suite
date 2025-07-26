import folium
from folium.plugins import MiniMap, Fullscreen, MarkerCluster
import pandas as pd
import logging
import osmnx as ox

logger = logging.getLogger(__name__)

GREEN_TAGS = [
    'forest', 'grass', 'recreation_ground', 'park', 'playground',
    'sports_centre', 'stadium', 'wood', 'grassland'
]

def create_base_map(city_center, key_locations, amenities, buildings, land_use, road_network, bbox):
    m = folium.Map(location=city_center, zoom_start=13, tiles='OpenStreetMap')
    logger.info("Creating base map...")

    #Key Locations (Red Markers)
    key_loc_group = folium.FeatureGroup(name="Key Locations", show=True)
    for location, coords in key_locations.items():
        folium.Marker(
            location=coords,
            popup=f"\U0001F4CD {location}",
            tooltip=location,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(key_loc_group)
    key_loc_group.add_to(m)

    #Amenities (Blue MarkerCluster)
    if amenities is not None and not amenities.empty:
        amenity_group = folium.FeatureGroup(name="Amenities", show=False)
        marker_cluster = MarkerCluster()
        for idx, row in amenities.iterrows():
            if 'geometry' in row and row.geometry is not None:
                geom = row.geometry
                # If Point, use directly; else use centroid
                if geom.geom_type == 'Point':
                    coords = [geom.y, geom.x]
                elif geom.geom_type in ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']:
                    centroid = geom.centroid
                    coords = [centroid.y, centroid.x]
                else:
                    continue
                folium.Marker(
                    location=coords,
                    popup=row.get('amenity', 'Amenity'),
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(marker_cluster)
        marker_cluster.add_to(amenity_group)
        amenity_group.add_to(m)

    # Buildings (Orange Polygons)
    if buildings is not None and not buildings.empty:
        folium.GeoJson(
            buildings,
            name="Buildings",
            style_function=lambda x: {'color': 'orange', 'weight': 1, 'fillOpacity': 0.2}
        ).add_to(m)

    # Green Spaces (Green Polygons)     
    if land_use is not None and not land_use.empty:
        mask = (
            land_use.get('landuse', pd.Series()).isin(GREEN_TAGS) |
            land_use.get('leisure', pd.Series()).isin(GREEN_TAGS) |
            land_use.get('natural', pd.Series()).isin(GREEN_TAGS)
        )
        green_gdf = land_use[mask]
        if not green_gdf.empty:
            folium.GeoJson(
                green_gdf,
                name="Green Spaces",
                style_function=lambda x: {'color': 'green', 'weight': 1, 'fillOpacity': 0.3}
            ).add_to(m)

    # Roads (Gray Lines) 
    if road_network is not None:
        nodes, edges = ox.graph_to_gdfs(road_network)
        folium.GeoJson(edges, name="Roads", style_function=lambda x: {
            'color': 'gray', 'weight': 2
        }).add_to(m)

    #Analysis Boundary (Blue Rectangle) 
    folium.Rectangle(
        bounds=[[bbox['south'], bbox['west']],
                [bbox['north'], bbox['east']]],
        color='blue',
        weight=2,
        fill=False,
        popup='Analysis Area (5km x 5km)'
    ).add_to(m)

    #Plugins and Layer Control
    MiniMap().add_to(m)
    Fullscreen().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

def add_green_legend(map_object, color_map, position="bottomright"):
    """
    Adds a legend to a Folium map for green space types.
    Args:
        map_object: folium.Map
        color_map: dict of {type: color}
        position: legend position (default: bottomright)
    """
    legend_html = f'''
    <div style="position: fixed; \
                {position}: 20px; z-index:9999; background-color: white; \
                border:2px solid grey; border-radius:6px; padding: 10px; font-size:14px;">
      <b>Green Space Types</b><br>
      {''.join([f'<div style="display: flex; align-items: center; margin-bottom: 4px;">'
                f'<span style="background:{color};width:18px;height:18px;display:inline-block;margin-right:8px;border:1px solid #888;"></span>'
                f'{label.title()}</div>' for label, color in color_map.items()])}
    </div>
    '''
    from branca.element import MacroElement
    from jinja2 import Template
    class Legend(MacroElement):
        def __init__(self, html):
            super().__init__()
            self._template = Template(f"""
                {{% macro html(this, kwargs) %}}
                {html}
                {{% endmacro %}}
            """)
    map_object.get_root().add_child(Legend(legend_html))
