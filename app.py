# app.py
import streamlit as st
import streamlit_folium as st_folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
import numpy as np
from datetime import datetime
import logging
import time
from typing import Optional, Tuple, Dict, Any

# Import modular components
from data.etl_pipeline import load_and_clean_data
from models.epidemic_simulation import sir_model
from models.demand_center_optimisation import greedy_p_center
from utils.ai_utilities import get_gemini_explanation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit UI Setup ---
st.set_page_config(
    layout="wide",
    page_title="Modular Epidemic Dashboard",
    page_icon="🇮🇳",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #e6e6e6;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    .alert-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---


# --- Enhanced Data Loading (Caching) ---
@st.cache_data(ttl=3600, show_spinner="🔄 Loading and processing data...")
def cached_load_and_clean_data():
    """Wrapper for ETL pipeline, utilizing Streamlit caching."""
    try:
        df_merged_full, beds_last_updated, df_merged_latest_states = load_and_clean_data()
        
        if df_merged_full.empty:
            logger.error("Failed to load data from ETL pipeline.")
            return None, None, None
            
        logger.info(f"Successfully loaded {len(df_merged_full)} records")
        
        # Ensure 'date' column is correctly formatted and 'active' is calculated
        for df in [df_merged_full, df_merged_latest_states]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.normalize()
            if 'active' not in df.columns:
                 df['active'] = (df['confirmed'] - df['recovered'] - df['deceased']).clip(lower=0)

        return df_merged_full, beds_last_updated, df_merged_latest_states
            
    except Exception as e:
        logger.error(f"Error in cached data loading: {e}")
        st.error(f"Failed to load data: {e}")
        return None, None, None

# Load data at the start of the script
df_merged_full, beds_last_updated, df_merged_latest_states = cached_load_and_clean_data()

# --- Initial UI setup and Data validation ---
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🇮🇳 AI-Powered Epidemic Early Warning & Response Dashboard")
st.caption("Proactive insights for health policymakers, powered by Google Gemini AI")
st.markdown('</div>', unsafe_allow_html=True)

if df_merged_full is None or df_merged_full.empty:
    st.error("Critical error: Unable to load essential data. Please check data sources and logs.")
    st.stop()
else:
    st.success(f"Global data loaded: {len(df_merged_full)} records. Dates from {df_merged_full['date'].min().strftime('%Y-%m-%d')} to {df_merged_full['date'].max().strftime('%Y-%m-%d')}.")
    # Display the first few rows and dtypes to ensure it looks correct
    # st.subheader("Debug: df_merged_full head")
    # st.dataframe(df_merged_full.head())
    # st.subheader("Debug: df_merged_full dtypes")
    # st.write(df_merged_full.dtypes)

if df_merged_latest_states is None or df_merged_latest_states.empty:
    st.error("Critical error: Unable to load latest state data.")
    st.stop()
else:
    st.success(f"Latest state data loaded: {len(df_merged_latest_states)} records.")
    # st.subheader("Debug: df_merged_latest_states head")
    # st.dataframe(df_merged_latest_states.head())
    # st.subheader("Debug: df_merged_latest_states dtypes")
    # st.write(df_merged_latest_states.dtypes)


# --- Sidebar ---
with st.sidebar:
    st.header("📊 Dashboard Controls")
    
    # Data refresh section
    st.subheader("Data Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col2:
        csv = df_merged_latest_states.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data",
            data=csv,
            file_name=f"epidemic_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Data status
    with st.expander("📈 Data Status", expanded=True):
        st.metric("Total Records", f"{len(df_merged_full):,}")
        st.metric("States and Union Territories ", df_merged_full['state_name'].nunique())
        st.info(f"**Last Updated:** {beds_last_updated}")
        
        # Check for missing coordinates
        missing_coords = df_merged_full[df_merged_full['latitude'].isna()]['state_name'].unique()
        if len(missing_coords) > 0:
            st.warning(f"⚠️ Missing coordinates: {', '.join(missing_coords)}")
        else:
            st.success("✅ All geographical data available")

# --- Main Dashboard Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard Overview", 
    "🦠 Epidemic Simulation", 
    "🏥 Resource Allocation",
    "📢 Community & Alerts"
])

# --- Tab 1: Dashboard Overview ---
# app.py

# ... (previous code) ...

# --- Tab 1: Dashboard Overview ---
with tab1:
    # --- Static Historical Trends Analysis ---
    st.subheader("📈 Historical Trends Analysis")
    st.info("View the epidemic's historical trend for each state.")

    # State selection
    available_states = sorted([s for s in df_merged_latest_states['state_name_standardized'].unique() if s != 'India'])
    if available_states:
        selected_state_hist = st.selectbox(
            "🔍 Select State for Historical Analysis:",
            available_states,
            index=available_states.index('Maharashtra') if 'Maharashtra' in available_states else 0,
            key='state_select_hist'
        )

        full_state_data = df_merged_latest_states[
            df_merged_latest_states['state_name_standardized'] == selected_state_hist
        ].sort_values('date').reset_index(drop=True)

        if not full_state_data.empty:
            st.info(f"Plotting {len(full_state_data)} data points from {full_state_data['date'].min().strftime('%Y-%m-%d')} to {full_state_data['date'].max().strftime('%Y-%m-%d')}")
            fig_hist = go.Figure()
            metrics = ['confirmed', 'active', 'recovered', 'deceased']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for metric, color in zip(metrics, colors):
                fig_hist.add_trace(go.Scatter(
                    x=full_state_data['date'],
                    y=full_state_data[metric],
                    mode='lines',
                    name=metric.title(),
                    line=dict(color=color, width=2)
                ))
            fig_hist.update_layout(
                title=f'Historical Trends in {selected_state_hist}',
                xaxis_title='Date',
                yaxis_title='Number of Cases',
                hovermode='x unified',
                height=500,
                showlegend=True,
                template='plotly_white',
                xaxis=dict(
                    range=[full_state_data['date'].min(), full_state_data['date'].max()]
                )
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning(f"No historical data available for {selected_state_hist}.")


      
    # State-wise Hotspot Analysis and Map (retained)
    st.subheader("🗺️ State-wise Hotspot Analysis")
    
    if not df_merged_latest_states.empty:
        metric_options = {
            'active': '🔴 Active Cases', 'confirmed': '📈 Total Confirmed',
            'recovered': '💚 Recovered', 'deceased': '⚫ Deaths', 'totalBeds': '🏥 Hospital Beds'
        }
        
        selected_metric = st.selectbox(
            "Select metric for analysis:",
            list(metric_options.keys()),
            format_func=lambda x: metric_options[x],
            key='state_metric_select'
        )

        state_data_filtered = df_merged_latest_states[
            df_merged_latest_states['state_name_standardized'] != 'India'
        ].copy()

        if not state_data_filtered.empty:
            # Bar chart
            fig_bar = px.bar(
                state_data_filtered.sort_values(selected_metric, ascending=True).tail(15), 
                x=selected_metric, y='state_name_original', orientation='h',
                title=f'Top 15 States by {metric_options[selected_metric]}',
                labels={'state_name_original': 'State/UT', selected_metric: 'Count'},
                color=selected_metric,
                color_continuous_scale='Reds' if selected_metric in ['active', 'deceased'] else 'Blues'
            )
            fig_bar.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Map Visualization
        st.subheader("🗺️ Geographic Distribution")
        map_data = state_data_filtered.dropna(subset=['latitude', 'longitude']).copy()
        
        if not map_data.empty:
            center_lat, center_lon = 20.5937, 78.9629
            m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")

            max_value = map_data[selected_metric].max()
            if max_value == 0: max_value = 1
            color_scheme = 'red' if selected_metric in ['active', 'deceased'] else 'green' if selected_metric == 'recovered' else 'blue'

            for idx, row in map_data.iterrows():
                normalized_size = (row[selected_metric] / max_value) * 20 + 5
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=normalized_size, color=color_scheme, fill=True, fillColor=color_scheme, fillOpacity=0.7,
                    popup=folium.Popup(
                        f"""<b>{row['state_name_original']}</b><br>{metric_options[selected_metric]}: {row[selected_metric]:,}<br>Active: {row['active']:,}<br>Hospital Beds: {row['totalBeds']:,}""",
                        max_width=300
                    ),
                    tooltip=f"{row['state_name_original']}: {row[selected_metric]:,}"
                ).add_to(m)

            st_folium.folium_static(m, width=1200, height=600)
            
            # Map insights
            with st.expander("🎯 Geographic Insights", expanded=False):
                top_3_states = map_data.nlargest(3, selected_metric)['state_name_original'].tolist()
                insights_prompt = f"""
                Based on the geographic distribution of {metric_options[selected_metric]} in India, 
                with the top 3 affected states being {', '.join(top_3_states)}, 
                provide brief strategic insights for resource allocation and intervention priorities.
                Keep under 100 words.
                """
                st.markdown(get_gemini_explanation(insights_prompt, cache_key=f"insights_{selected_metric}_{','.join(top_3_states)}"))

# --- Tab 2: Epidemic Simulation ---
with tab2:
    st.header("🦠 Epidemic Simulation & Policy Modeling")
    st.info("Run 'what-if' scenarios using the SIR epidemiological model to project future trends.")

    st.subheader("⚙️ Simulation Parameters")
    
    # Get initial values from data if available
    india_current = df_merged_latest_states[df_merged_latest_states['state_name_standardized'] == 'India']
    current_active = int(india_current['active'].iloc[0]) if not india_current.empty else 10000
    current_recovered = int(india_current['recovered'].iloc[0]) if not india_current.empty else 100000


    # Gemini explanations for tooltips
    total_population_help = get_gemini_explanation(
        "Explain what 'Total Population' means in an epidemic simulation and why it matters. Keep it concise.",
        cache_key="help_total_population"
    )
    initial_infected_help = get_gemini_explanation(
        "Explain what 'Initial Infected (I₀)' means in an epidemic simulation and why it matters. Keep it concise.",
        cache_key="help_initial_infected"
    )
    r_naught_help = get_gemini_explanation(
        "Explain what 'Basic Reproduction Number (R₀)' means in an epidemic simulation and why it matters. Keep it concise.",
        cache_key="help_r_naught"
    )
    recovery_rate_help = get_gemini_explanation(
        "Explain what 'Recovery Rate (γ) per day' means in an epidemic simulation and why it matters. Keep it concise.",
        cache_key="help_recovery_rate"
    )
    simulation_days_help = get_gemini_explanation(
        "Explain what 'Simulation Duration (days)' means in an epidemic simulation and why it matters. Keep it concise.",
        cache_key="help_simulation_days"
    )

    col_pop, col_init = st.columns(2)
    with col_pop:
        total_population = st.number_input(
            "Total Population",
            min_value=1000000,
            max_value=2000000000,
            value=1400000000,
            step=1000000,
            help=total_population_help
        )
    with col_init:
        initial_infected = st.number_input(
            "Initial Infected (I₀)",
            min_value=1,
            max_value=total_population//10,
            value=min(current_active, total_population//10),
            help=initial_infected_help
        )

    col_r0, col_gamma, col_days = st.columns(3)
    with col_r0:
        r_naught = st.slider(
            "Basic Reproduction Number (R₀)",
            min_value=0.1,
            max_value=6.0,
            value=1.5,
            step=0.1,
            help=r_naught_help
        )
    with col_gamma:
        recovery_rate = st.slider(
            "Recovery Rate (γ) per day",
            min_value=0.01,
            max_value=0.50,
            value=0.1,
            step=0.01,
            help=recovery_rate_help
        )
    with col_days:
        simulation_days = st.slider(
            "Simulation Duration (days)",
            min_value=30,
            max_value=730,
            value=180,
            step=30,
            help=simulation_days_help
        )

    # Calculate derived parameters
    beta = r_naught * recovery_rate
    initial_recovered = min(current_recovered, total_population - initial_infected)
    initial_susceptible = total_population - initial_infected - initial_recovered

    # --- AI-Powered Explanations for Simulation Parameters ---
    with st.expander("🤖 Explain Simulation Terms", expanded=False):
        st.write("Understand the key epidemiological terms used in this simulation.")
        explanation_choice = st.selectbox(
            "Select a term to explain:",
            [
                "Basic Reproduction Number (R₀)",
                "Recovery Rate (γ)",
                "Susceptible Population",
                "Infectious Population",
                "Recovered Population",
                "SIR Model"
            ],
            key='explanation_term_select'
        )

        # Generate prompt for the AI based on the selected term
        explanation_prompt = f"""
        Explain the epidemiological term '{explanation_choice}' in the context of an epidemic simulation,
        specifically for a public health dashboard.
        Keep the explanation concise, clear, and easy for a non-expert audience to understand.
        Focus on its role and significance in modeling disease spread.
        """
        
        # Use your existing get_gemini_explanation function
        st.markdown(get_gemini_explanation(explanation_prompt, cache_key=f"explanation_{explanation_choice}"))

    # Run simulation using the imported sir_model
    try:
        simulation_results = sir_model(
            total_population, initial_infected, initial_recovered, 
            beta, recovery_rate, simulation_days
        )
        
        # Plotting the results (retained)
        fig_sim = go.Figure()
        colors = {'Susceptible': '#1f77b4', 'Infectious': '#ff7f0e', 'Recovered': '#2ca02c'}
        for column in ['Susceptible', 'Infectious', 'Recovered']:
            fig_sim.add_trace(go.Scatter(
                x=simulation_results['Day'], y=simulation_results[column], mode='lines', name=column,
                line=dict(color=colors[column], width=3),
                hovertemplate=f'<b>{column}</b><br>Day: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
            ))
        
        fig_sim.update_layout(title='SIR Model Simulation Results', xaxis_title='Days', yaxis_title='Population', height=500, hovermode='x unified')
        st.plotly_chart(fig_sim, use_container_width=True)

        # Key metrics and AI explanations (retained)
        peak_infected_count = simulation_results['Infectious'].max()
        peak_infected_day = simulation_results['Infectious'].idxmax()
        total_affected = simulation_results['Recovered'].iloc[-1]

        st.subheader("📊 Key Simulation Outcomes")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Peak Infected Count", f"{int(peak_infected_count):,}")
        metric_col2.metric("Day of Peak Infection", f"Day {simulation_results.loc[peak_infected_day, 'Day']}")
        metric_col3.metric("Total Affected Population", f"{int(total_affected):,}")
        
        with st.expander("🤖 AI Policy Recommendations", expanded=True):
            policy_prompt = f"""
            Based on an SIR model simulation with the following parameters and outcomes:
            - R₀: {r_naught}, Recovery Rate: {recovery_rate}, Simulation Duration: {simulation_days} days
            - Peak Infected Count: {int(peak_infected_count):,} on day {simulation_results.loc[peak_infected_day, 'Day']}
            - Total Affected Population: {int(total_affected):,}
            Provide actionable policy recommendations for Indian health authorities.
            """
            st.markdown(get_gemini_explanation(policy_prompt, cache_key=f"policy_{r_naught}_{simulation_days}_{peak_infected_count}"))

    except Exception as e:
        st.error(f"An error occurred during simulation: {e}")
        st.warning("Please check input parameters or try again later.")

# --- Tab 3: Resource Allocation ---
with tab3:
    st.header("🏥 Resource Allocation Optimization")
    st.info("Use the P-Center model to identify optimal locations for new resource hubs.")

    # Prepare data for P-Center
    demand_centers_df = df_merged_latest_states[
        df_merged_latest_states['state_name_standardized'] != 'India'
    ].dropna(subset=['latitude', 'longitude']).copy()

    if demand_centers_df.empty:
        st.warning("No state data available for resource allocation.")
    else:
        st.subheader("⚙️ Allocation Parameters")
        num_facilities = st.slider(
            "Number of New Facilities to Place (P)",
            min_value=1, max_value=40, value=3
        )

        if st.button("🚀 Run P-Center Allocation", type="primary"):
            with st.spinner("Calculating optimal locations..."):
                # Use the imported greedy_p_center model
                chosen_facilities, max_dist, demand_centers_with_dist = greedy_p_center(
                    demand_centers_df, num_facilities
                )
            
            if not chosen_facilities.empty:
                st.success(f"**Optimization Complete!** Maximum distance to a new facility is **{max_dist:.2f} km**.")
                
                st.subheader("📍 Recommended Facility Locations")
                st.dataframe(chosen_facilities[['state_name_original', 'latitude', 'longitude']].rename(
                    columns={'state_name_original': 'State (Location)'}
                ))

                # Create map of results (retained)
                map_p_center = folium.Map(location=[22, 82], zoom_start=5, tiles="CartoDB positron")

                for _, row in demand_centers_with_dist.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']], radius=5, color='blue', fill=True, fill_color='blue',
                        tooltip=f"{row['state_name_original']}<br>Dist: {row['distance_to_closest_facility']:.1f} km"
                    ).add_to(map_p_center)

                for _, row in chosen_facilities.iterrows():
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        tooltip=f"**NEW FACILITY**<br>{row['state_name_original']}",
                        icon=folium.Icon(color='red', icon='star')
                    ).add_to(map_p_center)
                
                st_folium.folium_static(map_p_center, width=1200, height=600)

                with st.expander("🤖 AI Strategy Guide for Allocation", expanded=True):
                    pcenter_prompt = f"""
                    The P-Center algorithm was run to find {num_facilities} optimal locations for new resource hubs in India.
                    The model recommended placing them in: {', '.join(chosen_facilities['state_name_original'].tolist())}.
                    This resulted in a maximum travel distance of {max_dist:.2f} km for any state to reach a new hub.

                    Explain the strategic importance of this result for a national health logistics planner. 
                    What are the key advantages of this placement strategy?
                    Suggest the types of resources that should be prioritized for these new hubs.
                    """
                    st.markdown(get_gemini_explanation(pcenter_prompt, cache_key=f"pcenter_{num_facilities}_{max_dist}"))
            else:
                st.error("Could not determine optimal facility locations. Check input data.")

# --- Tab 4: Community & Alerts (PSA Generator) ---
with tab4:
    st.header("📢 AI-Powered Public Service Announcement (PSA) Generator")
    st.info("Quickly generate clear and effective public health messages for different audiences.")
    
    st.subheader("📝 Create Your PSA")
    
    psa_topic = st.selectbox("Select a Topic", ("Vaccination Drive", "Hygiene Practices", "New Variant Alert", "Travel Advisory", "Mental Health Support"))
    psa_audience = st.selectbox("Select Target Audience", ("General Public", "Healthcare Workers", "Rural Communities", "Urban Youth", "Elderly Population"))
    psa_tone = st.select_slider("Select Message Tone", options=["Informative", "Reassuring", "Urgent", "Empathetic", "Direct"], value="Informative")
    psa_points = st.text_area("Key Points to Include (optional, one per line)")
    
    if st.button("✍️ Generate PSA", type="primary"):
        with st.spinner("Crafting your message with AI..."):
            psa_prompt = f"""
            Act as a public health communications expert. Generate a Public Service Announcement (PSA) for India.
            **Topic:** {psa_topic}, **Target Audience:** {psa_audience}, **Desired Tone:** {psa_tone}, **Key Points to Include:**\n{psa_points}
            """
            generated_psa = get_gemini_explanation(psa_prompt, cache_key=f"psa_{psa_topic}_{psa_audience}_{psa_tone}_{hash(psa_points)}")
            
            st.subheader("✅ Your Generated PSA")
            st.markdown(generated_psa)