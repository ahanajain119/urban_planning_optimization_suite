import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing import get_city_stats

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

    #Map Placeholder
    st.subheader("City Map")
    st.info("City map will be loaded here once we process OpenStreetMap data")
        
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
    st.markdown('<div class="sub-header ">Traffic flow & Congestion Analysis</div>', unsafe_allow_html=True)
    st.info("Traffic analysis will be implemented in the next phase")

elif page == "Enviornmental Impact":
    st.markdown('<div class="sub-header">Environmental IQuality and Clean Space</div>', unsafe_allow_html=True)
    st.info("Environmental impact analysis will be implemented in the next phase")

elif page == "Economic Optimization":
    st.markdown('<div class="sub-header">Cost Benefit Analysis & Budget Optimization</div>', unsafe_allow_html=True)
    st.info("Economic optimization will be implemented in the next phase")

elif page == "Settings":
    st.markdown('<div class="sub-header">Settings & Configuration</div>', unsafe_allow_html=True)

    st.subheader("City Parameters")
    city_size = st.slider("City Size (km²)", 10, 100, 25)
    population = st.slider("Population", 10000, 100000, 45000)
    
    st.subheader("Optimization Weights")
    traffic_weight = st.slider("Traffic Efficiency Weight", 0.0, 1.0, 0.3)
    env_weight = st.slider("Environmental Weight", 0.0, 1.0, 0.3)
    economic_weight = st.slider("Economic Weight", 0.0, 1.0, 0.4)
    st.info(f"Current weights: Traffic={traffic_weight}, Environment={env_weight}, Economic={economic_weight}")

# Footer
st.markdown("---")
st.markdown("Urban Planning Optimization Suite built using Streamlit, GeoPandas, and OSMnx")










