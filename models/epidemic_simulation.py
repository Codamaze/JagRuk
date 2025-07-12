import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def sir_model(N: int, I0: int, R0: int, beta: float, gamma: float, num_days: int) -> pd.DataFrame:
    """
    SIR model simulation.
    
    N: Total population
    I0: Initial infected
    R0: Initial recovered
    beta: Transmission rate
    gamma: Recovery rate
    num_days: Duration of simulation
    """
    # Validate inputs
    if N <= 0 or I0 < 0 or R0 < 0 or beta < 0 or gamma <= 0:
        raise ValueError("Invalid parameters for SIR model")

    # Ensure initial conditions are valid
    if I0 + R0 > N:
        logger.warning("Initial infected + recovered exceeds population. Adjusting.")
        I0 = min(I0, N)
        R0 = min(R0, N - I0)

    S = [max(0, N - I0 - R0)]
    I = [I0]
    R = [R0]
    
    dt = 0.1  # Smaller time step for better numerical stability
    steps_per_day = int(1.0 / dt)
    
    # Simulation loop using numerical integration
    for day in range(num_days):
        for step in range(steps_per_day):
            current_S = max(0, S[-1])
            current_I = max(0, I[-1])
            current_R = max(0, R[-1])

            # If the epidemic is essentially over, stop the spread and maintain population counts
            if current_I < 1:
                new_S, new_I, new_R = current_S, current_I, current_R
            else:
                # SIR differential equations
                dSdt = -beta * current_S * current_I / N
                dIdt = beta * current_S * current_I / N - gamma * current_I
                dRdt = gamma * current_I

                # Euler method integration
                new_S = current_S + dSdt * dt
                new_I = current_I + dIdt * dt
                new_R = current_R + dRdt * dt

            S.append(new_S)
            I.append(new_I)
            R.append(new_R)

    # Downsample to daily values for plotting
    daily_S = []
    daily_I = []
    daily_R = []
    daily_days = []

    for d in range(num_days + 1):
        idx_at_end_of_day = d * steps_per_day
        
        if idx_at_end_of_day < len(S):
            daily_S.append(S[idx_at_end_of_day])
            daily_I.append(I[idx_at_end_of_day])
            daily_R.append(R[idx_at_end_of_day])
        else:
            # If simulation ended early, use the last calculated values
            daily_S.append(S[-1])
            daily_I.append(I[-1])
            daily_R.append(R[-1])
            
        daily_days.append(d)
            
    return pd.DataFrame({
        'Day': daily_days,
        'Susceptible': daily_S,
        'Infectious': daily_I,
        'Recovered': daily_R
    })