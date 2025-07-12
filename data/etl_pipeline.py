import requests
import pandas as pd
from typing import Tuple, Dict, Optional
import logging
from datetime import datetime
import time

# Import configuration
from config import COVID_DATA_URL, BEDS_DATA_URL, STATE_CODE_TO_NAME, ROOTNET_TO_STANDARD_NAME, STATE_COORDINATES

# Configure logging
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

def fetch_with_retry(url: str, max_retries: int = 3, timeout: int = 10) -> Optional[Dict]:
    """Fetch data from URL with retry logic."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching data from {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error on attempt {attempt + 1}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error during fetch: {e}")
            break
        time.sleep(1) # Wait briefly before retry
    return None

def validate_covid_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean COVID data (Transformation phase)."""
    if df.empty:
        raise DataValidationError("COVID data is empty")
    
    # Check for required columns
    required_cols = ['confirmed', 'recovered', 'deceased', 'date', 'state_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    
    # Data quality checks (e.g., negative values)
    numeric_cols = ['confirmed', 'recovered', 'deceased', 'tested', 'vaccinated1', 'vaccinated2']
    for col in numeric_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                logger.warning(f"Negative values found in {col} - replacing with 0")
                df[col] = df[col].clip(lower=0)
    
    # Ensure 'date' column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

    return df

def validate_beds_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean hospital beds data (Transformation phase)."""
    if df.empty:
        logger.warning("Hospital beds data is empty")
        return df
    
    # Check for negative bed counts
    bed_cols = ['ruralBeds', 'urbanBeds', 'totalBeds']
    for col in bed_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                df[col] = df[col].clip(lower=0)
    
    return df

def process_covid_data(covid_data: Dict) -> pd.DataFrame:
    """Processes raw COVID timeseries data into a DataFrame."""
    all_states = []
    for state_code, state_data in covid_data.items():
        if state_code not in STATE_CODE_TO_NAME:
            continue
            
        dates = state_data.get('dates', {})
        if not dates:
            continue
            
        try:
            df = pd.DataFrame.from_dict(dates, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Extract numeric columns safely
            for col in ['confirmed', 'recovered', 'deceased', 'tested', 'vaccinated1', 'vaccinated2']:
                df[col] = df['total'].apply(
                    lambda x: x.get(col, 0) if isinstance(x, dict) else 0
                )
            
            df = df.reset_index().rename(columns={'index': 'date'})
            df['state_code'] = state_code
            df['state_name'] = STATE_CODE_TO_NAME[state_code]
            all_states.append(df)
            
        except Exception as e:
            logger.warning(f"Error processing data for {state_code}: {e}")
            continue
    
    if not all_states:
        raise DataValidationError("No valid state data found after processing")
    
    df_covid = pd.concat(all_states, ignore_index=True)
    df_covid = df_covid.fillna(0)
    
    # Ensure numeric types
    numeric_cols = ['confirmed', 'recovered', 'deceased', 'tested', 'vaccinated1', 'vaccinated2']
    for col in numeric_cols:
        df_covid[col] = pd.to_numeric(df_covid[col], errors='coerce').fillna(0)

    return validate_covid_data(df_covid)

def process_beds_data(beds_data: Dict) -> Tuple[pd.DataFrame, str]:
    """Processes hospital beds data and returns a cleaned DataFrame and last update time."""
    if not beds_data:
        logger.warning("Beds data dictionary is empty.")
        return pd.DataFrame(), 'N/A'

    df_beds = pd.DataFrame(beds_data['data']['regional'])
    beds_last_updated = beds_data['data'].get('lastOriginUpdate', 'N/A')
    
    df_beds = validate_beds_data(df_beds)

    # State name mapping for beds data
    if not df_beds.empty:
        df_beds['state_name'] = df_beds['state'].map(ROOTNET_TO_STANDARD_NAME)
        df_beds = df_beds.dropna(subset=['state_name'])
        
        # Combine Dadra & Nagar Haveli and Daman & Diu
        dn_dd = df_beds[df_beds['state_name'] == 'Dadra and Nagar Haveli and Daman and Diu']
        if len(dn_dd) > 1:
            combined = dn_dd[['ruralHospitals', 'ruralBeds', 'urbanHospitals', 
                           'urbanBeds', 'totalHospitals', 'totalBeds']].sum()
            combined['state_name'] = 'Dadra and Nagar Haveli and Daman and Diu'
            df_beds = df_beds[df_beds['state_name'] != 'Dadra and Nagar Haveli and Daman and Diu']
            df_beds = pd.concat([df_beds, pd.DataFrame([combined])], ignore_index=True)
            
    return df_beds, beds_last_updated

def merge_and_finalize_data(df_covid: pd.DataFrame, df_beds: pd.DataFrame) -> pd.DataFrame:
    """Merges COVID and beds data, adds coordinates and derived metrics."""
    if not df_covid.empty and not df_beds.empty:
        df_merged = df_covid.merge(df_beds, on='state_name', how='left')
    else:
        df_merged = df_covid.copy()
    
    # Fill missing values and ensure bed columns are numeric
    bed_cols = ['ruralHospitals', 'ruralBeds', 'urbanHospitals', 'urbanBeds', 
                'totalHospitals', 'totalBeds']
    for col in bed_cols:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)
    
    # Calculate derived metrics
    df_merged['active'] = (df_merged['confirmed'] - df_merged['recovered'] - 
                           df_merged['deceased']).clip(lower=0)
    
    # Add coordinates
    df_merged['latitude'] = df_merged['state_name'].map(
        lambda x: STATE_COORDINATES.get(x, {}).get('latitude')
    )
    df_merged['longitude'] = df_merged['state_name'].map(
        lambda x: STATE_COORDINATES.get(x, {}).get('longitude')
    )
    
    return df_merged

def load_and_clean_data() -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    """
    Main ETL pipeline function.
    Fetches raw data, processes, cleans, and merges it.
    Returns: (df_merged_full, beds_last_updated, df_merged_latest_states)
    """
    logger.info("Starting data loading and cleaning process.")
    
    try:
        # 1. Extraction
        covid_raw = fetch_with_retry(COVID_DATA_URL)
        beds_raw = fetch_with_retry(BEDS_DATA_URL)
        
        if not covid_raw:
            logger.error("Failed to fetch COVID data.")
            return pd.DataFrame(), "N/A", pd.DataFrame()
        
        # 2. Transformation
        df_covid = process_covid_data(covid_raw)
        df_beds, beds_last_updated = process_beds_data(beds_raw)
        
        # 3. Merging and Finalization
        df_merged_full = merge_and_finalize_data(df_covid, df_beds)
        
        # Create latest states dataset
        df_merged_latest_states = (df_merged_full.sort_values('date')
                                   .groupby('state_name')
                                   .tail(1)
                                   .reset_index(drop=True))
        
        df_merged_latest_states['state_name_original'] = df_merged_latest_states['state_name']
        df_merged_latest_states.rename(columns={'state_name': 'state_name_standardized'}, inplace=True)
        
        logger.info("Data loading complete.")
        return df_merged_full, beds_last_updated, df_merged_latest_states
        
    except DataValidationError as e:
        logger.error(f"Data validation failed: {e}")
        return pd.DataFrame(), "N/A", pd.DataFrame()
    except Exception as e:
        logger.error(f"Error in ETL pipeline: {e}")
        return pd.DataFrame(), "N/A", pd.DataFrame()