import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
import osmnx as ox


from src.data_processing import get_city_stats, MysoreDataLoader  
from src.visualization import create_base_map
# Set page configuration
st.set_page_config(page_title="Urban Planning Optimizer", 
                   layout="wide",
                    initial_sidebar_state="expanded"
)

#custom CSS for styling
st.markdown("""
<style>
            .mainheader {
                text-align: center;
                margin-bottom: 2rem;
                font-size: 3rem;
                font-weight: bold;
                color: #2E86AB;
            }
            .sub-header {
                font-size: 1.5rem;
                color: #A23B72;
                margin: 1rem 0;
            }
            .metric-box {
                background-color: #F18F01;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem;
                color: white;
                text-align: center;
            }
</style>
""", unsafe_allow_html=True)

#Main Title
st.markdown("<div class='mainheader'>Urban Planning Optimizer</div>", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.title("Navigation")
city = st.sidebar.selectbox("Select City", ["Mysore"], index=0)
page= st.sidebar.radio("Select Analysis Mode", 
                       ["City Overview", "Traffic Analysis", "Enviornmental Impact", "Economic Optimization", "Settings"])

# Load city statistics
stats = get_city_stats(city)
city_population = stats['population']
city_area = stats['area']
green_space_pct = stats['green_space_pct']
traffic_score = stats['traffic_score']


if stats is None or city_population == 0:
    st.warning("City data could not be loaded. Please try again later.")

#Main content 
if page == "City Overview":
    st.markdown('<div class="sub-header">City Overview & Basic Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-box">Population<br><h2>{city_population:,}</h2></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-box">Area<br><h2>{city_area:,}</h2></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-box">Green Space<br><h2>{green_space_pct:,}</h2></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="metric-box">Traffic Score<br><h2>{traffic_score:,}</h2></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    #Map 
    st.subheader("City Map")
    try:
        loader = MysoreDataLoader()
        loader.load_mysore_data(save_to_file=False)
        m = create_base_map(
            loader.city_center,
            loader.key_locations,
            loader.amenities,
            loader.buildings,
            loader.land_use,
            loader.road_network,
            loader.bbox
        )
        folium_static(m)
    except Exception as e:
        st.error(f"Could not load map: {e}")
        
    # Placeholder for basic charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Land Use Distribution")
        land_use = stats.get('land_use', None)
        if land_use:
            labels = list(land_use.keys())
            sizes = list(land_use.values())

            # Define custom colors: grass=green, residential=yellow, others use Set3
            base_colors = plt.get_cmap('Set3').colors
            color_map = []
            for label in labels:
                if label == "grass":
                    color_map.append("#4CAF50")  # green
                elif label == "residential":
                    color_map.append("#FFD600")  # yellow
                else:
                    color_map.append(base_colors[len(color_map) % len(base_colors)])

            fig, ax = plt.subplots(figsize=(6, 6))
            wedges, texts = ax.pie(
                sizes,
                labels=None,
                autopct=None,
                startangle=140,
                colors=color_map,
                pctdistance=0.8,
                textprops={'fontsize': 14, 'weight': 'bold', 'color': 'black'},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                shadow=True
            )

            # Draw a white circle at the center for donut look
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig.gca().add_artist(centre_circle)

            # Prepare legend labels with percentages
            total = sum(sizes)
            legend_labels = [
                f"{label}: {size/total*100:.1f}%" for label, size in zip(labels, sizes)
            ]

            # Add legend outside the pie
            ax.legend(wedges, legend_labels, title="Land Use", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("Land use data not available.")

    with col2:
                st.subheader("Population Density")
                # Estimate density: population / area, shown as a single value for now
                if city_area > 0:
                    density = city_population / city_area
                    st.metric("People per km²", f"{density:,.0f}")
                else:
                    st.info("Area data not available.")          

elif page == "Traffic Analysis":
    st.markdown('<div class="sub-header ">Traffic Flow & Congestion Analysis</div>', unsafe_allow_html=True)

    try:
        loader = MysoreDataLoader()
        loader.load_mysore_data(save_to_file=False)
        m = folium.Map(location=loader.city_center, zoom_start=13, tiles='OpenStreetMap')

        # Add road network as gray lines
        if loader.road_network is not None:
            nodes, edges = ox.graph_to_gdfs(loader.road_network)
            folium.GeoJson(
                edges,
                name="Roads",
                style_function=lambda x: {'color': 'gray', 'weight': 2}
            ).add_to(m)
        folium.LayerControl().add_to(m)
        folium_static(m)
    except Exception as e:
        st.error(f"Could not load road network map: {e}")

    # --- Basic Traffic Stats ---
    st.subheader("Network Statistics")
    if loader.road_network is not None:
        num_edges = len(loader.road_network.edges())
        num_nodes = len(loader.road_network.nodes())
        degrees = dict(loader.road_network.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Road Segments", f"{num_edges:,}")
        col2.metric("Total Intersections", f"{num_nodes:,}")
        col3.metric("Avg. Node Degree", f"{avg_degree:.2f}")
    else:
        st.info("No road network data available.")

elif page == "Enviornmental Impact":
    st.markdown('<div class="sub-header">Environmental Quality and Green Space Analysis</div>', unsafe_allow_html=True)
    try:
        loader = MysoreDataLoader()
        loader.load_mysore_data(save_to_file=False)
        from src.data_processing import create_square_grid, calculate_green_space_per_zone, calculate_pollution_score_per_zone, calculate_green_space_pct
        import plotly.express as px
        import pandas as pd
        # --- Ecological Quality Metrics ---
        # 1. Calculate total green area and % green area
        land_use = loader.land_use
        bbox = loader.bbox
        total_area_km2 = 25  # 5x5 km
        GREEN_TAGS = [
            'forest', 'grass', 'recreation_ground', 'park', 'playground',
            'sports_centre', 'stadium', 'wood', 'grassland'
        ]
        # Filter green polygons
        mask = (
            land_use.get('landuse', pd.Series()).isin(GREEN_TAGS) |
            land_use.get('leisure', pd.Series()).isin(GREEN_TAGS) |
            land_use.get('natural', pd.Series()).isin(GREEN_TAGS)
        )
        green_gdf = land_use[mask]
        if not green_gdf.empty:
            green_gdf_proj = green_gdf.to_crs(epsg=32643)
            total_green_area_km2 = green_gdf_proj.geometry.area.sum() / 1e6
        else:
            total_green_area_km2 = 0
        green_pct = calculate_green_space_pct(land_use, total_area_km2)
        # 2. Pollution score (proxy: more roads, less green)
        # Road density: total road length / area
        nodes, edges = ox.graph_to_gdfs(loader.road_network)
        edges_proj = edges.to_crs(epsg=32643)
        total_road_km = edges_proj.length.sum() / 1000
        road_density = total_road_km / total_area_km2 if total_area_km2 > 0 else 0
        # Proxy pollution: higher with more road density, lower with more green
        # Normalize: pollution_score = 100 * (road_density / max_road_density) * (1 - green_pct/100)
        # Assume max_road_density = 10 km/km² for normalization
        max_road_density = st.sidebar.slider(
            "Max Road Density for Pollution Score (km/km²)",
            min_value=2.0, max_value=20.0, value=10.0, step=0.5,
            help="Adjusts the normalization for the pollution score. Higher values mean more tolerance for road density."
        )
        pollution_score = 100 * (road_density / max_road_density) * (1 - green_pct/100)
        pollution_score = min(max(pollution_score, 0), 100)
        # --- Display Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Green Area (km²)", f"{total_green_area_km2:.2f}")
        col2.metric("% Green Area", f"{green_pct:.2f}%")
        col3.metric("Total Road Length (km)", f"{total_road_km:.2f}")
        col4.metric("Pollution Score", f"{pollution_score:.1f} / 100")
        # --- Info Box ---
        st.info("""
        The ecological quality and pollution score shown here are based on a simple proxy model: pollution increases with road density and decreases with green space. This is a demonstration and can be extended with more factors (e.g., traffic, industry, air quality sensors) in the future.
        """)
        # --- Folium Map: Green Areas by Type ---
        st.subheader("Green Areas Map (by Type)")
        m = folium.Map(location=loader.city_center, zoom_start=13, tiles='OpenStreetMap')
        color_map = {
            'forest': '#228B22',
            'grass': '#7CFC00',
            'recreation_ground': '#90EE90',
            'park': '#006400',
            'playground': '#32CD32',
            'sports_centre': '#98FB98',
            'stadium': '#2E8B57',
            'wood': '#556B2F',
            'grassland': '#B2FF66'
        }
        for tag in GREEN_TAGS:
            tag_gdf = green_gdf[green_gdf.get('landuse', '') == tag]
            if tag_gdf.empty:
                tag_gdf = green_gdf[green_gdf.get('leisure', '') == tag]
            if tag_gdf.empty:
                tag_gdf = green_gdf[green_gdf.get('natural', '') == tag]
            if not tag_gdf.empty:
                folium.GeoJson(
                    tag_gdf,
                    name=tag,
                    style_function=lambda x, color=color_map.get(tag, '#4CAF50'): {
                        'fillColor': color,
                        'color': color,
                        'weight': 1,
                        'fillOpacity': 0.5
                    },
                    tooltip=folium.GeoJsonTooltip(fields=["landuse", "leisure", "natural"], aliases=["Landuse", "Leisure", "Natural"])
                ).add_to(m)
        from src.visualization import add_green_legend
        add_green_legend(m, color_map, position="bottomright")
        folium.LayerControl().add_to(m)
        from streamlit_folium import folium_static
        folium_static(m)
        # --- Bonus: Bar Chart of Green Space Types ---
        st.subheader("Green Space Type Breakdown")
        # Count by type (landuse, leisure, natural)
        type_counts = {}
        for tag in GREEN_TAGS:
            count = (
                (green_gdf.get('landuse', pd.Series()) == tag).sum() +
                (green_gdf.get('leisure', pd.Series()) == tag).sum() +
                (green_gdf.get('natural', pd.Series()) == tag).sum()
            )
            if count > 0:
                type_counts[tag] = count
        if type_counts:
            df_types = pd.DataFrame({"Type": list(type_counts.keys()), "Count": list(type_counts.values())})
            fig = px.bar(df_types, x="Type", y="Count", color="Type", title="Green Space Types", color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No green space type data available.")
        # --- Existing grid and pollution maps ---
        st.markdown("---")
        grid_gdf = create_square_grid(loader.bbox, n_rows=5, n_cols=5)
        if grid_gdf.crs.to_string() != "EPSG:4326":
            grid_gdf = grid_gdf.to_crs(epsg=4326)
        green_stats_gdf = calculate_green_space_per_zone(loader.land_use, grid_gdf)
        pollution_gdf = calculate_pollution_score_per_zone(loader.road_network, grid_gdf, green_stats_gdf)
        merged_gdf = green_stats_gdf.merge(pollution_gdf[['zone_id', 'pollution_score']], on='zone_id')
        from matplotlib import cm, colors
        # Green Space Map by Zone
        st.subheader("Green Space by Zone (5x5 Grid)")
        m_grid = folium.Map(location=loader.city_center, zoom_start=13, tiles='OpenStreetMap')
        def green_style_function(feature):
            pct = feature['properties']['green_pct']
            color = cm.get_cmap('RdYlGn')(pct/100)
            hex_color = colors.to_hex(color, keep_alpha=False)
            return {
                'fillColor': hex_color,
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.7
            }
        folium.GeoJson(
            merged_gdf,
            name="Green Space Grid",
            style_function=green_style_function,
            tooltip=folium.GeoJsonTooltip(fields=["zone_id", "green_pct", "pollution_score"],
                                          aliases=["Zone:", "Green %:", "Pollution:"],
                                          localize=True)
        ).add_to(m_grid)
        folium.LayerControl().add_to(m_grid)
        folium_static(m_grid)
        # --- Export to CSV for zone pollution table ---
        csv = merged_gdf[['zone_id', 'green_pct', 'pollution_score']].sort_values('zone_id').reset_index(drop=True).to_csv(index=False)
        st.download_button(
            label="Export Zone Pollution Table to CSV",
            data=csv,
            file_name="zone_pollution_table.csv",
            mime="text/csv"
        )
        # Zone Statistics Table
        st.subheader("Zone Statistics")
        st.dataframe(merged_gdf[['zone_id', 'green_pct', 'pollution_score']].sort_values('zone_id').reset_index(drop=True))
        # Pollution Score Map
        st.subheader("Pollution Score by Zone (Red = Higher)")
        m2 = folium.Map(location=loader.city_center, zoom_start=13, tiles='OpenStreetMap')
        def pollution_style_function(feature):
            score = feature['properties']['pollution_score']
            norm = min(max((score+2)/4, 0), 1)
            color = cm.get_cmap('RdYlGn_r')(norm)
            hex_color = colors.to_hex(color, keep_alpha=False)
            return {
                'fillColor': hex_color,
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.7
            }
        folium.GeoJson(
            merged_gdf,
            name="Pollution Grid",
            style_function=pollution_style_function,
            tooltip=folium.GeoJsonTooltip(fields=["zone_id", "green_pct", "pollution_score"],
                                          aliases=["Zone:", "Green %:", "Pollution:"],
                                          localize=True)
        ).add_to(m2)
        folium.LayerControl().add_to(m2)
        folium_static(m2)
    except Exception as e:
        st.error(f"Could not load environmental analysis: {e}")

elif page == "Economic Optimization":
    st.markdown('<div class="sub-header">Cost Benefit Analysis & Budget Optimization</div>', unsafe_allow_html=True)
    loader = MysoreDataLoader()
    loader.load_mysore_data(save_to_file=False)
    # --- Data Inputs ---
    road_network = loader.road_network
    land_use = loader.land_use
    amenities = loader.amenities
    city_population = stats.get('population', 0)
    # --- Cost Parameters (from session_state if available) ---
    unit_costs = st.session_state.get('unit_costs', None)
    if unit_costs:
        ROAD_COST_PER_KM = unit_costs['road']
        GREEN_COST_PER_HA = unit_costs['green']
        AMENITY_COSTS = unit_costs['amenities']
    else:
        ROAD_COST_PER_KM = 10_000_000
        GREEN_COST_PER_HA = 500_000
        AMENITY_COSTS = {
            'hospital': 50_000_000,
            'school': 30_000_000,
            'park': 20_000_000,
            'university': 40_000_000,
            'college': 25_000_000,
            'stadium': 60_000_000,
            'sports_centre': 15_000_000,
            'playground': 10_000_000,
            'library': 8_000_000,
            'clinic': 12_000_000,
            'fire_station': 20_000_000,
            'police': 20_000_000,
            'marketplace': 10_000_000,
            'bank': 5_000_000,
            'pharmacy': 3_000_000,
            'cafe': 2_000_000,
            'restaurant': 3_000_000,
            'fuel': 5_000_000
        }
    # --- Calculate Road Cost ---
    nodes, edges = ox.graph_to_gdfs(road_network)
    edges_proj = edges.to_crs(epsg=32643)
    total_road_km = edges_proj.length.sum() / 1000
    road_cost = total_road_km * ROAD_COST_PER_KM
    # --- Calculate Green Space Cost ---
    GREEN_TAGS = [
        'forest', 'grass', 'recreation_ground', 'park', 'playground',
        'sports_centre', 'stadium', 'wood', 'grassland'
    ]
    mask = (
        land_use.get('landuse', pd.Series()).isin(GREEN_TAGS) |
        land_use.get('leisure', pd.Series()).isin(GREEN_TAGS) |
        land_use.get('natural', pd.Series()).isin(GREEN_TAGS)
    )
    green_gdf = land_use[mask]
    if not green_gdf.empty:
        green_gdf_proj = green_gdf.to_crs(epsg=32643)
        total_green_area_km2 = green_gdf_proj.geometry.area.sum() / 1e6
    else:
        total_green_area_km2 = 0
    total_green_area_ha = total_green_area_km2 * 100  # 1 km² = 100 ha
    green_cost = total_green_area_ha * GREEN_COST_PER_HA
    # --- Calculate Amenity Cost ---
    amenity_breakdown = {}
    total_amenity_cost = 0
    if amenities is not None and not amenities.empty:
        for amenity_type, cost in AMENITY_COSTS.items():
            count = (amenities.get('amenity', pd.Series()) == amenity_type).sum()
            if count > 0:
                amenity_breakdown[amenity_type] = count * cost
                total_amenity_cost += count * cost
    # --- Cost Breakdown ---
    cost_breakdown = {
        'Roads': road_cost,
        'Green Spaces': green_cost,
        'Amenities': total_amenity_cost
    }
    total_cost = sum(cost_breakdown.values())
    cost_per_capita = total_cost / city_population if city_population > 0 else 0
    # --- Sidebar Budget Widget ---
    st.sidebar.markdown("---")
    budget = st.sidebar.slider(
        "Set Infrastructure Budget (₹M)",
        min_value=500, max_value=2000, value=1000, step=10
    ) * 1_000_000
    # --- Progress Bar and Budget Status ---
    st.subheader("Budget Comparison")
    pct = min(total_cost / budget, 1.0)
    st.progress(pct, text=f"{total_cost/1_000_000:.1f}M / {budget/1_000_000:.1f}M")
    if total_cost > budget:
        st.warning(f"Total infrastructure cost exceeds budget by ₹{(total_cost-budget)/1_000_000:.1f}M!")
    else:
        st.success(f"Total infrastructure cost is within budget.")
    # --- Key Economic Metrics ---
    st.subheader("Key Economic Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Total Cost (₹)", f"{total_cost:,.0f}")
    col2.metric("Cost per Capita (₹)", f"{cost_per_capita:,.0f}")
    # --- Cost Breakdown Chart ---
    st.subheader("Infrastructure Cost Breakdown")
    import plotly.express as px
    breakdown_df = pd.DataFrame({
        'Category': list(cost_breakdown.keys()),
        'Cost': list(cost_breakdown.values())
    })
    # Export to CSV button
    csv_cost = breakdown_df.to_csv(index=False)
    st.download_button(
        label="Export Cost Breakdown Table to CSV",
        data=csv_cost,
        file_name="cost_breakdown_table.csv",
        mime="text/csv"
    )
    fig = px.pie(breakdown_df, names='Category', values='Cost', title='Cost Contribution by Category', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
    # --- Amenity Breakdown Chart ---
    if amenity_breakdown:
        st.subheader("Amenity Cost Breakdown")
        amenity_df = pd.DataFrame({
            'Amenity': list(amenity_breakdown.keys()),
            'Cost': list(amenity_breakdown.values())
        })
        fig2 = px.bar(amenity_df, x='Amenity', y='Cost', title='Amenity Cost Contribution', text_auto='.2s')
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Settings":
    st.markdown('<div class="sub-header">Settings & Configuration</div>', unsafe_allow_html=True)

    st.subheader("City Parameters")
    city_size = st.slider("City Size (km²)", 10, 100, 25)
    population = st.slider("Population", 10000, 100000, 45000)

    st.subheader("Unit Costs (Customize)")
    if 'unit_costs' not in st.session_state:
        st.session_state.unit_costs = {
            'road': 10_000_000,
            'green': 500_000,
            'amenities': {
                'hospital': 50_000_000,
                'school': 30_000_000,
                'park': 20_000_000,
                'university': 40_000_000,
                'college': 25_000_000,
                'stadium': 60_000_000,
                'sports_centre': 15_000_000,
                'playground': 10_000_000,
                'library': 8_000_000,
                'clinic': 12_000_000,
                'fire_station': 20_000_000,
                'police': 20_000_000,
                'marketplace': 10_000_000,
                'bank': 5_000_000,
                'pharmacy': 3_000_000,
                'cafe': 2_000_000,
                'restaurant': 3_000_000,
                'fuel': 5_000_000
            }
        }
    st.session_state.unit_costs['road'] = st.number_input("Road Cost per km (₹)", min_value=1_000_000, max_value=50_000_000, value=st.session_state.unit_costs['road'], step=500_000)
    st.session_state.unit_costs['green'] = st.number_input("Green Space Cost per hectare (₹)", min_value=100_000, max_value=5_000_000, value=st.session_state.unit_costs['green'], step=50_000)
    st.markdown("**Amenity Unit Costs (₹ each)**")
    for amenity, default in st.session_state.unit_costs['amenities'].items():
        st.session_state.unit_costs['amenities'][amenity] = st.number_input(f"{amenity.title()} Cost", min_value=500_000, max_value=100_000_000, value=default, step=500_000, key=f"amenity_cost_{amenity}")

    st.subheader("Optimization Weights")
    col1, col2 = st.columns(2)
    with col1:
        economic_weight = st.slider("Economic Weight", 0.0, 1.0, 0.5, step=0.05, key="economic_weight")
    with col2:
        env_weight = st.slider("Environmental Weight", 0.0, 1.0, 0.5, step=0.05, key="env_weight")
    # Normalize weights to sum to 1
    total = economic_weight + env_weight
    if total > 0:
        st.session_state.optim_weights = {
            'economic': economic_weight / total,
            'environmental': env_weight / total
        }
    else:
        st.session_state.optim_weights = {'economic': 0.5, 'environmental': 0.5}
    st.info(f"Current weights: Economic={st.session_state.optim_weights['economic']:.2f}, Environment={st.session_state.optim_weights['environmental']:.2f}")

# Add Urban Layout Optimization page
if 'Urban Layout Optimization' not in ["City Overview", "Traffic Analysis", "Enviornmental Impact", "Economic Optimization", "Settings"]:
    pass  # If you want to add to sidebar, update the selectbox/radio options
elif page == "Urban Layout Optimization":
    st.markdown('<div class="sub-header">Urban Layout Optimization (Multi-Objective)</div>', unsafe_allow_html=True)
    st.info("This module uses evolutionary algorithms to suggest better city layouts by balancing traffic, environment, cost, and accessibility.")
    # --- Objective Weights ---
    st.subheader("Objective Weights")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        w_travel = st.slider("Travel Time Weight", 0.0, 1.0, 0.25, step=0.05, key="w_travel")
    with col2:
        w_env = st.slider("Environmental Quality Weight", 0.0, 1.0, 0.25, step=0.05, key="w_env")
    with col3:
        w_cost = st.slider("Economic Cost Weight", 0.0, 1.0, 0.25, step=0.05, key="w_cost")
    with col4:
        w_access = st.slider("Accessibility Weight", 0.0, 1.0, 0.25, step=0.05, key="w_access")
    # Normalize
    total = w_travel + w_env + w_cost + w_access
    if total > 0:
        weights = [w_travel/total, w_env/total, w_cost/total, w_access/total]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
    st.session_state.optim_weights = {
        'travel': weights[0],
        'environmental': weights[1],
        'economic': weights[2],
        'access': weights[3]
    }
    # --- Run Optimization Button ---
    from src.optimization import run_nsga2, individuals_to_grids
    import plotly.express as px
    import pandas as pd
    if st.button("Run Optimization", type="primary"):
        with st.spinner("Running multi-objective optimization (NSGA-II)..."):
            unit_costs = st.session_state.get('unit_costs', {'road': 10_000_000, 'green': 5_000_000})
            top5, best = run_nsga2(pop_size=50, ngen=20, unit_costs=unit_costs)
            best_grid = np.array(best).reshape((5,5))
            # --- Best Layout Visualization ---
            st.subheader("Best Layout Visualization")
            color_map = {"residential": "#FFD600", "commercial": "#1976D2", "industrial": "#757575", "green": "#43A047"}
            fig = px.imshow([[color_map[cell] for cell in row] for row in best_grid],
                            color_continuous_scale=list(color_map.values()),
                            aspect="auto", title="Optimized City Layout (5x5)")
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
            # --- Top 5 Layouts Table ---
            st.subheader("Top 5 Layouts Table")
            data = []
            for i, ind in enumerate(top5):
                travel, green, cost, access = ind.fitness.values
                data.append({
                    "Rank": i+1,
                    "Travel Time": f"{travel:.2f}",
                    "Green %": f"{(-green)*100:.1f}",
                    "Cost (₹M)": f"{cost/1e6:.1f}",
                    "Access Score": f"{access:.2f}"
                })
            st.dataframe(pd.DataFrame(data))
            # --- Current vs Optimized Layout Comparison ---
            st.subheader("Current vs Optimized Layout Comparison")
            # Placeholder: current layout is random
            current_grid = np.random.choice(list(color_map.keys()), size=(5,5))
            fig2 = px.imshow([[color_map[cell] for cell in row] for row in current_grid],
                             color_continuous_scale=list(color_map.values()),
                             aspect="auto", title="Current Layout (5x5)")
            fig2.update_xaxes(showticklabels=False)
            fig2.update_yaxes(showticklabels=False)
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Current Layout**")
                st.plotly_chart(fig2, use_container_width=True)
            with colB:
                st.markdown("**Optimized Layout**")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("Best Layout Visualization")
        st.info("Color-coded map of best layout will appear here.")
        st.subheader("Top 5 Layouts Table")
        st.info("Table of top 5 layouts and their objective scores will appear here.")
        st.subheader("Current vs Optimized Layout Comparison")
        st.info("Side-by-side comparison of current and optimized layouts will appear here.")

# Footer
st.markdown("---")
st.markdown("Urban Planning Optimization Suite built using Streamlit, GeoPandas, and OSMnx")










