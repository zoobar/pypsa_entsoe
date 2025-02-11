# Conversion factors
GASOIL_TONNE_MWH = 11.94
COAL_TONNE_MWH = 8.14

# Emission factors in tCO2/MWh
EMISSION_FACTORS = {
    'lignite': 0.4,
    'coal': 0.34, 
    'gas': 0.201,
    'oil': 0.28,
    'CCGT': 0.201,
    'OCGT': 0.201
} 

THERMAL_FUELS = ['CCGT', 'OCGT', 'coal', 'lignite', 'oil']