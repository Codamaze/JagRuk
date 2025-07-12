COVID_DATA_URL = "https://data.incovid19.org/v4/min/timeseries.min.json"
BEDS_DATA_URL = "https://api.rootnet.in/covid19-in/hospitals/beds"

# State coordinates (Example for India)
STATE_COORDINATES = {
    "Andaman and Nicobar Islands": {"latitude": 11.7401, "longitude": 92.6586},
    "Andhra Pradesh": {"latitude": 15.9129, "longitude": 79.7400},
    "Arunachal Pradesh": {"latitude": 28.2180, "longitude": 94.7278},
    "Assam": {"latitude": 26.2006, "longitude": 92.9376},
    "Bihar": {"latitude": 25.0961, "longitude": 85.3131},
    "Chandigarh": {"latitude": 30.7333, "longitude": 76.7794},
    "Chhattisgarh": {"latitude": 21.2787, "longitude": 81.8661},
    "Dadra and Nagar Haveli and Daman and Diu": {"latitude": 20.4283, "longitude": 72.9338},
    "Delhi": {"latitude": 28.7041, "longitude": 77.1025},
    "Goa": {"latitude": 15.2993, "longitude": 74.1240},
    "Gujarat": {"latitude": 22.2587, "longitude": 71.1924},
    "Haryana": {"latitude": 29.0588, "longitude": 76.0856},
    "Himachal Pradesh": {"latitude": 31.1048, "longitude": 77.1734},
    "Jammu and Kashmir": {"latitude": 33.7782, "longitude": 76.5762},
    "Jharkhand": {"latitude": 23.6102, "longitude": 85.2799},
    "Karnataka": {"latitude": 15.3173, "longitude": 75.7139},
    "Kerala": {"latitude": 10.8505, "longitude": 76.2711},
    "Ladakh": {"latitude": 34.1526, "longitude": 77.5770},
    "Lakshadweep": {"latitude": 10.5667, "longitude": 72.6417},
    "Madhya Pradesh": {"latitude": 22.9734, "longitude": 78.6569},
    "Maharashtra": {"latitude": 19.7515, "longitude": 75.7139},
    "Manipur": {"latitude": 24.6637, "longitude": 93.9063},
    "Meghalaya": {"latitude": 25.4670, "longitude": 91.9934},
    "Mizoram": {"latitude": 23.1645, "longitude": 92.9376},
    "Nagaland": {"latitude": 26.1584, "longitude": 94.5624},
    "Odisha": {"latitude": 20.9517, "longitude": 85.0985},
    "Puducherry": {"latitude": 11.9416, "longitude": 79.8083},
    "Punjab": {"latitude": 31.1471, "longitude": 75.3412},
    "Rajasthan": {"latitude": 27.0238, "longitude": 74.2179},
    "Sikkim": {"latitude": 27.5330, "longitude": 88.5122},
    "Tamil Nadu": {"latitude": 11.1271, "longitude": 78.6569},
    "Telangana": {"latitude": 18.1124, "longitude": 79.0193},
    "Tripura": {"latitude": 23.9408, "longitude": 91.9882},
    "Uttar Pradesh": {"latitude": 26.8467, "longitude": 80.9462},
    "Uttarakhand": {"latitude": 30.0668, "longitude": 79.0193},
    "West Bengal": {"latitude": 22.9868, "longitude": 87.8550},
    "India": {"latitude": 20.5937, "longitude": 78.9629}
}

# Mapping of state codes to names
STATE_CODE_TO_NAME = {
    'AN': 'Andaman and Nicobar Islands', 'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh', 'AS': 'Assam', 'BR': 'Bihar',
    'CH': 'Chandigarh', 'CT': 'Chhattisgarh', 'DL': 'Delhi',
    'DN': 'Dadra and Nagar Haveli and Daman and Diu', 'GA': 'Goa',
    'GJ': 'Gujarat', 'HP': 'Himachal Pradesh', 'HR': 'Haryana',
    'JH': 'Jharkhand', 'JK': 'Jammu and Kashmir', 'KA': 'Karnataka',
    'KL': 'Kerala', 'LA': 'Ladakh', 'LD': 'Lakshadweep',
    'MH': 'Maharashtra', 'ML': 'Meghalaya', 'MN': 'Manipur',
    'MP': 'Madhya Pradesh', 'MZ': 'Mizoram', 'NL': 'Nagaland',
    'OR': 'Odisha', 'PB': 'Punjab', 'PY': 'Puducherry',
    'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TG': 'Telangana',
    'TN': 'Tamil Nadu', 'TR': 'Tripura', 'TT': 'India',
    'UP': 'Uttar Pradesh', 'UT': 'Uttarakhand', 'WB': 'West Bengal'
}

# Mapping for hospital beds data source (Rootnet) to our standard names
ROOTNET_TO_STANDARD_NAME = {
    "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
    "Andhra Pradesh": "Andhra Pradesh",
    "Arunachal Pradesh": "Arunachal Pradesh",
    "Assam": "Assam", "Bihar": "Bihar", "Chhattisgarh": "Chhattisgarh",
    "Goa": "Goa", "Gujarat": "Gujarat", "Haryana": "Haryana",
    "Himachal Pradesh": "Himachal Pradesh", "Jammu & Kashmir": "Jammu and Kashmir",
    "Jharkhand": "Jharkhand", "Karnataka": "Karnataka", "Kerala": "Kerala",
    "Madhya Pradesh": "Madhya Pradesh", "Maharashtra": "Maharashtra",
    "Manipur": "Manipur", "Meghalaya": "Meghalaya", "Mizoram": "Mizoram",
    "Nagaland": "Nagaland", "Odisha": "Odisha", "Punjab": "Punjab",
    "Rajasthan": "Rajasthan", "Sikkim": "Sikkim", "Tamil Nadu": "Tamil Nadu",
    "Telangana": "Telangana", "Tripura": "Tripura", "Uttar Pradesh": "Uttar Pradesh",
    "Uttarakhand": "Uttarakhand", "West Bengal": "West Bengal",
    "Chandigarh": "Chandigarh", "Delhi": "Delhi", "Lakshadweep": "Lakshadweep",
    "Puducherry": "Puducherry",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "INDIA": "India"
}