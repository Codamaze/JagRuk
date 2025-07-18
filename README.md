# 🇮🇳 JagRuk: AI-Powered Epidemic Early Warning & Response Dashboard

## Overview

**JagRuk** is an advanced, modular dashboard designed to empower health policymakers, researchers, and emergency responders with real-time epidemic intelligence for India. By integrating live data, epidemiological modeling, optimization algorithms, and the power of Google Gemini AI, JagRuk delivers actionable insights, resource allocation strategies, and effective public health messaging—all in one place.

---

## 🚀 Motivation & Impact

India faces unique challenges in epidemic management due to its vast population, diverse geography, and resource constraints. Timely, data-driven decisions can save thousands of lives. JagRuk bridges the gap between raw data and actionable policy by:

- Providing a unified, interactive platform for epidemic monitoring and forecasting.
- Enabling rapid scenario analysis and resource planning.
- Leveraging AI to generate clear, trustworthy explanations and public health messages.

**Goal:** Empower decision-makers to act early, allocate resources optimally, and communicate effectively during health crises.

---

## 🧩 Key Features

### 1. Dashboard Overview
- Visualize historical COVID-19 and epidemic trends for every Indian state and UT.
- Interactive time-series graphs and state-wise hotspot bar charts.
- Geographic mapping of cases, recoveries, deaths, and hospital beds.
- AI-generated strategic insights for resource allocation.

### 2. Epidemic Simulation
- Run SIR (Susceptible-Infectious-Recovered) model simulations with custom parameters (R₀, recovery rate, etc.).
- Visualize projected epidemic curves and key inflection points.
- AI-powered, layman-friendly explanations of epidemiological terms and simulation results.

### 3. Resource Allocation Optimization
- Use the P-Center model to identify optimal locations for new resource hubs.
- Minimize maximum travel distance for critical resources.
- Interactive maps and AI-generated strategy guides for logistics planning.

### 4. Community & Alerts
- Instantly generate Public Service Announcements (PSAs) tailored to different audiences and tones using Google Gemini AI.
- Support for multiple topics: vaccination, hygiene, new variants, travel advisories, mental health, and more.

---

## 🏗️ Architecture & Tech Stack

- **Frontend:** Streamlit (UI), Plotly (charts), Folium (maps)
- **Backend:** Python (Pandas, Numpy, custom models)
- **AI Integration:** Google Gemini API (for explanations, insights, and PSA generation)
- **Optimization & Simulation:**
  - SIR Model (epidemic simulation)
  - Greedy P-Center (resource hub placement)
- **ETL Pipeline:** Automated data loading, cleaning, and merging from public APIs and resource datasets


## 📊 Example Use Cases

- **Health Ministry:** Monitor real-time trends, simulate interventions, and plan resource allocation for outbreak response.
- **Researchers:** Analyze historical data, test epidemiological hypotheses, and visualize model outcomes.
- **Public Health Communicators:** Instantly generate clear, targeted PSAs for different audiences and scenarios.

---

## 🤖 AI Integration Details

- **Google Gemini API** powers:
  - Layman-friendly explanations of epidemiological concepts and simulation results.
  - Strategic insights for resource allocation and intervention priorities.
  - Automatic generation of PSAs in multiple tones and for diverse audiences.

---

## 📄 License

MIT License

---

*Made with ❤️ for public health, data science, and a safer India.*