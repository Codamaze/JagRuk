import pandas as pd
import numpy as np
from geopy.distance import great_circle
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def greedy_p_center(demand_centers_df: pd.DataFrame, 
                    p_facilities: int,
                    potential_facility_locations_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Greedy P-Center algorithm to find optimal facility locations.
    
    demand_centers_df: DataFrame of locations requiring service (states/districts)
    p_facilities: Number of facilities to locate
    potential_facility_locations_df: Optional DataFrame of potential locations (defaults to demand_centers)
    """
    
    if demand_centers_df.empty:
        logger.warning("Demand centers DataFrame is empty.")
        return pd.DataFrame(), float('inf'), pd.DataFrame()

    if potential_facility_locations_df is None:
        potential_facility_locations_df = demand_centers_df.copy()

    # Clean data (ensure lat/lon exist)
    demand_centers_clean = demand_centers_df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True).copy()
    potential_locations_clean = potential_facility_locations_df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True).copy()

    if demand_centers_clean.empty or potential_locations_clean.empty:
        logger.error("Insufficient valid coordinates for P-Center calculation.")
        return pd.DataFrame(), float('inf'), pd.DataFrame()

    n_demand = len(demand_centers_clean)
    n_potential = len(potential_locations_clean)
    
    if p_facilities <= 0 or p_facilities > n_potential:
        logger.warning(f"Invalid number of facilities requested: {p_facilities}")
        p_facilities = min(p_facilities, n_potential)

    # Precompute distance matrix (in km)
    distance_matrix = np.zeros((n_demand, n_potential))
    
    demand_coords = list(zip(demand_centers_clean['latitude'], demand_centers_clean['longitude']))
    potential_coords = list(zip(potential_locations_clean['latitude'], potential_locations_clean['longitude']))
    
    for i in range(n_demand):
        for j in range(n_potential):
            # Calculate great-circle distance between demand center i and potential facility j
            distance_matrix[i, j] = great_circle(demand_coords[i], potential_coords[j]).km

    # Greedy selection process
    selected_facilities = []
    min_distances = np.full(n_demand, float('inf'))

    for facility_num in range(min(p_facilities, n_potential)):
        best_facility_idx = -1
        best_max_distance = float('inf')

        for candidate_idx in range(n_potential):
            if candidate_idx in selected_facilities:
                continue

            # Calculate the impact of adding this candidate facility on the max distance
            temp_min_distances = np.minimum(min_distances, distance_matrix[:, candidate_idx])
            max_distance = np.max(temp_min_distances)

            # Choose the candidate that minimizes the maximum distance
            if max_distance < best_max_distance:
                best_max_distance = max_distance
                best_facility_idx = candidate_idx

        if best_facility_idx != -1:
            selected_facilities.append(best_facility_idx)
            min_distances = np.minimum(min_distances, distance_matrix[:, best_facility_idx])

    # Prepare results
    if selected_facilities:
        chosen_facilities_df = potential_locations_clean.iloc[selected_facilities].copy()
        chosen_facilities_df['is_facility'] = True
        final_max_distance = np.max(min_distances)
        
        demand_centers_clean['distance_to_closest_facility'] = min_distances
        
        return chosen_facilities_df, final_max_distance, demand_centers_clean
    else:
        return pd.DataFrame(), float('inf'), demand_centers_clean