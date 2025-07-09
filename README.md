# Urban Planning Optimization Suite

## Project Overview
The Urban Planning Optimization Suite is a toolkit for analyzing, visualizing, and optimizing urban environments using open data. It is designed to help urban planners, researchers, and students explore city infrastructure, amenities, land use, and more, with a focus on the city of Mysore (India) as a case study.

## Data Pipeline
1. **Data Loading**: Fetches spatial data (roads, buildings, amenities, land use, POIs) from OpenStreetMap using OSMnx.
2. **Data Processing**: Cleans, structures, and computes statistics on the data (see `src/data_processing.py`).
3. **Visualization**: Generates interactive maps and visual summaries using Folium and Streamlit (see `src/visualization.py`).
4. **Simulation/Optimization**: (Planned) Modules for simulating urban scenarios and optimizing city layouts (see `src/simulation.py` and `src/optimization.py`).

## Dependencies
- **streamlit**: Web app framework for interactive dashboards
- **pandas, numpy**: Data manipulation and analysis
- **matplotlib, seaborn, plotly**: Data visualization
- **geopandas**: Geospatial data handling
- **folium, streamlit_folium**: Interactive maps
- **osmnx, networkx**: Downloading and analyzing street networks
- **scikit-learn, scipy, deap**: Machine learning, optimization, and scientific computing
- **requests, beautifulsoup4**: Web requests and HTML parsing

See `requirements.txt` for exact versions.

## Usage Instructions
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Urban\ Planning\ Optimization\ Suite
   ```
2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```
5. **Explore the dashboard**
   - Load Mysore city data
   - Visualize roads, buildings, amenities, and green spaces
   - View city statistics and (in future) run simulations/optimizations

## Project Structure
- `src/data_processing.py`: Data loading, cleaning, and statistics for Mysore
- `src/visualization.py`: Map and plotting functions (Folium, etc.)
- `src/simulation.py`: (Planned) Urban scenario simulation logic
- `src/optimization.py`: (Planned) Optimization algorithms for urban planning
- `streamlit_app.py`: Main Streamlit dashboard app
- `data/`: Stores raw and processed data

## Future Work
- Add simulation models for urban growth, traffic, or land use
- Implement optimization routines for city planning scenarios
- Expand to support more cities and data sources

---
For questions or contributions, please open an issue or pull request!
