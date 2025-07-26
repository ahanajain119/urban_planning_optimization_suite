import random
from deap import base, creator, tools, algorithms
import numpy as np
import networkx as nx
import osmnx as ox
from shapely.geometry import Point

ZONE_TYPES = ["residential", "commercial", "industrial", "green"]
GRID_SIZE = 5
N_ZONES = GRID_SIZE * GRID_SIZE

# --- Fitness and Individual ---
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))  # All objectives minimized
creator.create("Individual", list, fitness=creator.FitnessMulti)

def create_individual():
    return creator.Individual([random.choice(ZONE_TYPES) for _ in range(N_ZONES)])

def individual_to_grid(ind):
    return np.array(ind).reshape((GRID_SIZE, GRID_SIZE))

def get_grid_centroids(bbox, n_rows=5, n_cols=5):
    minx, miny, maxx, maxy = bbox['west'], bbox['south'], bbox['east'], bbox['north']
    width = (maxx - minx) / n_cols
    height = (maxy - miny) / n_rows
    centroids = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = minx + (j + 0.5) * width
            y = miny + (i + 0.5) * height
            centroids.append((y, x))  # (lat, lon)
    return centroids

# --- Fitness Function ---
def evaluate_layout(ind, amenities=None, road_network=None, unit_costs=None, bbox=None):
    grid = individual_to_grid(ind)
    # 1. Get grid cell centroids (lat, lon)
    if bbox is None:
        # Default to Mysore bbox
        bbox = {'north': 12.3051 + 0.03, 'south': 12.3051 - 0.03, 'east': 76.6553 + 0.03, 'west': 76.6553 - 0.03}
    centroids = get_grid_centroids(bbox, GRID_SIZE, GRID_SIZE)
    # 2. Prepare amenity locations by type
    essential_types = ["school", "hospital", "market", "park"]
    amenity_points = {t: [] for t in essential_types}
    if amenities is not None:
        for t in essential_types:
            matches = amenities[amenities["amenity"] == t] if "amenity" in amenities.columns else amenities[amenities["leisure"] == t]
            for _, row in matches.iterrows():
                geom = row.geometry
                if geom.geom_type == "Point":
                    amenity_points[t].append((geom.y, geom.x))
                else:
                    centroid = geom.centroid
                    amenity_points[t].append((centroid.y, centroid.x))
    # 3. Map centroids to nearest road network node
    node_ids = []
    if road_network is not None:
        for lat, lon in centroids:
            try:
                node = ox.nearest_nodes(road_network, lon, lat)
            except Exception:
                node = None
            node_ids.append(node)
    else:
        node_ids = [None] * len(centroids)
    # 4. Travel Time: For each residential cell, compute shortest path to nearest amenity (school/hospital/market)
    travel_times = []
    for idx, zone in enumerate(ind):
        if zone != "residential":
            continue
        node = node_ids[idx]
        min_time = None
        for t in ["school", "hospital", "market"]:
            for pt in amenity_points.get(t, []):
                try:
                    amenity_node = ox.nearest_nodes(road_network, pt[1], pt[0])
                    length = nx.shortest_path_length(road_network, node, amenity_node, weight="length")
                    # Assume 30 km/h = 0.5 km/min
                    time = (length / 1000) / 0.5  # minutes
                    if min_time is None or time < min_time:
                        min_time = time
                except Exception:
                    # Fallback: Euclidean distance
                    d = np.sqrt((centroids[idx][0] - pt[0])**2 + (centroids[idx][1] - pt[1])**2) * 111  # deg to km
                    time = d / 0.5
                    if min_time is None or time < min_time:
                        min_time = time
        if min_time is not None:
            travel_times.append(min_time)
    travel_time = np.mean(travel_times) if travel_times else 60.0
    # 5. Green score: % green zones
    green_score = np.sum(grid == "green") / N_ZONES
    # 6. Cost: sum by zone type
    cost = 0
    cost += np.sum(grid == "green") * (unit_costs.get("green", 5_000_000) if unit_costs else 5_000_000)
    cost += np.sum(grid == "residential") * (unit_costs.get("road", 10_000_000) if unit_costs else 10_000_000)
    cost += (np.sum(grid == "commercial") + np.sum(grid == "industrial")) * (unit_costs.get("commercial", 15_000_000) if unit_costs else 15_000_000)
    # Add cost for hospitals/schools if present in grid (assume one per cell if amenity exists in that cell)
    for idx, zone in enumerate(ind):
        if zone in ["residential", "commercial", "industrial", "green"]:
            for t in ["hospital", "school"]:
                for pt in amenity_points.get(t, []):
                    # If amenity is within cell (roughly, within 1/2 cell width)
                    cell_lat, cell_lon = centroids[idx]
                    d = np.sqrt((cell_lat - pt[0])**2 + (cell_lon - pt[1])**2) * 111
                    if d < 0.5:  # ~0.5 km
                        cost += unit_costs.get(t, 50_000_000) if unit_costs else 50_000_000
    # 7. Accessibility: For all cells, average distance to nearest essential service (school, hospital, park)
    access_dists = []
    for idx, (lat, lon) in enumerate(centroids):
        min_dist = None
        for t in ["school", "hospital", "park"]:
            for pt in amenity_points.get(t, []):
                try:
                    node = node_ids[idx]
                    amenity_node = ox.nearest_nodes(road_network, pt[1], pt[0])
                    length = nx.shortest_path_length(road_network, node, amenity_node, weight="length")
                    dist = length / 1000  # km
                except Exception:
                    dist = np.sqrt((lat - pt[0])**2 + (lon - pt[1])**2) * 111
                if min_dist is None or dist < min_dist:
                    min_dist = dist
        if min_dist is not None:
            access_dists.append(min_dist)
    access_score = np.mean(access_dists) if access_dists else 10.0
    return (travel_time, -green_score, cost, access_score)

# --- Toolbox Setup ---
def get_toolbox(amenities=None, road_network=None, unit_costs=None):
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_layout, amenities=amenities, road_network=road_network, unit_costs=unit_costs)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(ZONE_TYPES)-1, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    return toolbox

# --- Main Optimization Function ---
def run_nsga2(pop_size=100, ngen=40, amenities=None, road_network=None, unit_costs=None):
    toolbox = get_toolbox(amenities, road_network, unit_costs)
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.7, mutpb=0.3, ngen=ngen, stats=stats, halloffame=hof, verbose=False)
    # Sort by first objective (travel time)
    sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0])
    top5 = sorted_pop[:5]
    best = top5[0]
    return top5, best

# --- Utility to convert individuals to grid layouts ---
def individuals_to_grids(individuals):
    return [individual_to_grid(ind) for ind in individuals]
