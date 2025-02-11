import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from thermal_help import read_eraa_capacities_country_year
# The power system model
import pypsa

# Two support static classes
from metenergy_data import *
from pypsa_support import *

ENTSOE_API_KEY="874a992e-d415-4891-b909-cf974b1b9fde"


LIST_COUNTRIES = ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT',
'LV', 'LT', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK', 'UK']



# Which period? 
START        = np.datetime64('2021-01-01 00:00')
END          = np.datetime64('2021-12-31 23:00')

YEAR         = np.datetime_as_string(START, unit='Y')
FUTURE_YEAR= '2025'
SIMNAME = '202105'





# Data downloaded from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS)

# 2m air temperature
tmp = pd.read_csv('H_ERA5_ECMW_T639_TA-_0002m_Euro_NUT0_S197901010000_E202410312300_INS_TIM_01h_NA-_noc_org_NA_NA---_NA---_NA---.csv', 
skiprows=52,
parse_dates=True,
index_col='Date')

# Surface downwelling shortwave radiation
ssr = pd.read_csv('H_ERA5_ECMW_T639_GHI_0000m_Euro_NUT0_S197901010000_E202410312300_INS_TIM_01h_NA-_noc_org_NA_NA---_NA---_NA---.csv', 
skiprows=52,
parse_dates=True,
index_col='Date')

# Wind-onshore capacity factor
won = pd.read_csv('H_ERA5_ECMW_T639_WON_0100m_Euro_NUT0_S197901010000_E202410312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
skiprows=52, parse_dates=True,
index_col='Date')

# Wind-offshore capacity factor
wof = pd.read_csv('H_ERA5_ECMW_T639_WOF_0100m_Euro_MAR0_S197901010000_E202410312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
skiprows=52, parse_dates=True,
index_col='Date')

# Hydro power generation rivers
ror = pd.read_csv('H_ERA5_ECMW_T639_HRO_NA---_Euro_NUT0_S197901010000_E202410312300_CFR_TIM_01d_NA-_noc_org_NA_NA---_NA---_StRnF.csv', 
skiprows=61,
parse_dates=True,
index_col='Date')

# Solar radiation data can have some values smaller than zero (~1e-17) that can create issues in the modelling
ssr[ssr < 0] = 0

# Duplicating the data for Greece using the country code `GR` in addition to `EL` to be consistent with ENTSO-E data
tmp['GR'] = tmp['EL']
won['GR'] = won['EL']
wof['GR'] = wof['EL']
ror['GR'] = ror['EL']
ssr['GR'] = ssr['EL']


other_map = {
    'AT':{'Biomass': 1.0},
    'SE':{'Biomass':0.3, 'Fossil Oil': 0.2, 'Fossil Gas': 0.5},
    'PL':{'Biomass':0.6, 'Fossil Oil': 0.4},
    'DE':{'Fossil Oil': 1.0},
    'FR':{ 'Fossil Gas': 1.0},
    'IE':{'Wind Onshore': 1.0},
    'IT':{'Solar': 1.0},
    'SK':{'Biomass': 1.0},
    'UK':{'Fossil Gas': 1.0}
    }


NSTEPS = (END - START).astype('timedelta64[h]').astype(int)
print(f'{NSTEPS} steps to simulate')
n = pypsa.Network(snapshots = list(range(NSTEPS)))


# get fuel prices


from units import GASOIL_TONNE_MWH, EMISSION_FACTORS, COAL_TONNE_MWH, THERMAL_FUELS

gas_price = 57#40
coal_price = 105#107
eua_price = 82.94#63
eur_usd = 1.03#1.05
lignite_price = 3.1#3.1
oil_price = 762.833#762.833


#can you generate a function to calculate generation cost from following fuel prices?
fuel_prices_df = pd.DataFrame(
    {
        'gas_price': [gas_price] * NSTEPS,  # EUR/MWh
        'coal_price': [coal_price] * NSTEPS,  # EUR/ton
        'eua_price': [eua_price] * NSTEPS,  # EUR/tCO2
        'eur_usd': [eur_usd] * NSTEPS,
        'lignite_price': [lignite_price] * NSTEPS,  # EUR/MWh
        'oil_price': [oil_price] * NSTEPS  # EUR/ton
    }
)

def calculate_marginal_cost(fuel, fuel_prices_df, efficiency):
    """Calculate marginal cost for each fuel type based on fuel prices and efficiency.
    
    Args:
        fuel: String indicating fuel type ('coal', 'gas', 'lignite', 'oil')
        fuel_prices_df: DataFrame containing fuel and carbon prices
        efficiency: Efficiency of the generator
        
    Returns:
        Marginal cost in EUR/MWh
    """
    if fuel in ['coal', 'Coal', 'Hard Coal', 'Hard coal']:
        # Convert coal price from EUR/ton to EUR/MWh using efficiency
        fuel_cost = fuel_prices_df['coal_price'] / COAL_TONNE_MWH / efficiency / fuel_prices_df['eur_usd']
        carbon_cost = fuel_prices_df['eua_price'] * EMISSION_FACTORS['coal'] / efficiency
        
    elif fuel in ['CCGT', 'OCGT', 'gas', 'Natural gas', 'Gas']:
        fuel_cost = fuel_prices_df['gas_price'] / efficiency
        carbon_cost = fuel_prices_df['eua_price'] * EMISSION_FACTORS['gas'] / efficiency
        
    elif fuel == 'lignite':
        fuel_cost = fuel_prices_df['lignite_price'] / efficiency
        carbon_cost = fuel_prices_df['eua_price'] * EMISSION_FACTORS['lignite'] / efficiency
        
    elif fuel == 'oil':
        # Convert oil price from EUR/ton to EUR/MWh using efficiency and conversion factor
        fuel_cost = fuel_prices_df['oil_price'] / GASOIL_TONNE_MWH / efficiency / fuel_prices_df['eur_usd']
        carbon_cost = fuel_prices_df['eua_price'] * EMISSION_FACTORS['oil'] / efficiency
        
    else:
        return 0
        
    return fuel_cost + carbon_cost



import numpy as np
import pandas as pd

# === HELPER FUNCTIONS ===

def get_demand(country, start, end):
    """Obtain and adjust the demand timeseries."""
    dem = metenergy_data.get_demand_entsoe(
        country, 
        timeline=pd.date_range(start=start, end=end, tz='UTC'),
        MY_API_KEY=ENTSOE_API_KEY
    )
    dem.index = dem.index.tz_convert(None)
    return dem

def get_wind_cf(country, dem):
    """Extract onshore (and, if available, offshore) wind capacity factors."""
    won_cf = won.loc[dem.index, country]
    wof_cf = wof.loc[dem.index, country] if country in wof.columns else None
    return won_cf, wof_cf

def get_solar_cf(country, dem, year):
    """Compute the solar capacity factor for a given country and year."""
    sp = (
        metenergy_data
        .get_PV_cf(
            tmp=tmp.loc[year, country],
            ssr=ssr.loc[year, country].groupby(level=0).last()  # type: ignore
        )
        .loc[dem.index]
    )
    return sp

def get_generation_capacity(country, year):
    """Download and filter ENTSO-E generation capacity data."""
    cap = metenergy_data.get_capacity_entsoe(zone=country, year=int(year), MY_API_KEY=ENTSOE_API_KEY)
    # Keep only technologies with >50 MW capacity.
    cap = cap.loc[:, (cap > 50).any(axis=0)]
    return cap

def adjust_capacities(cap, country):
    """Apply country‐specific corrections to capacity data."""
    if country in ['CZ', 'DE', 'PL']:
        cap = cap.drop(columns='Hydro Water Reservoir')
    elif country == 'SK':
        cap['Hydro Run-of-river and poundage'] = (
            cap['Hydro Run-of-river and poundage'] + cap['Hydro Water Reservoir']
        )
        cap = cap.drop(columns='Hydro Water Reservoir')
    return cap

def replace_other(cap, country):
    """Replace the generic 'Other' category with country-specific mappings."""
    if 'Other' in cap.columns and cap['Other'].values[0] > 500:
        for key, factor in other_map.get(country, {}).items():
            if key in cap.columns:
                cap[key] += cap['Other'] * factor
            else:
                cap[key] = cap['Other'] * factor
        cap = cap.drop(columns=['Other'])
    return cap

# replace future thermal and renewables capacities with ERAA data
    #overwrite thermal and renewable capacities 
    #Get ERAA capacities for future year
    
def replace_future_capacities(country, year):    
    future_caps = read_eraa_capacities_country_year(country, year)
    
    # Map ERAA names to ENTSO-E names
    eraa_to_entsoe = {
        'Hard Coal': 'Fossil Hard coal',
        'Lignite': 'Fossil Brown coal/Lignite',
        'Nuclear': 'Nuclear', 
        'Biomass': 'Biomass',
        'Gas': 'Fossil Gas',
        'Wind Onshore': 'Wind Onshore',
        'Wind Offshore': 'Wind Offshore',
        'Solar (Photovoltaic)': 'Solar'
    }
    
    # Check if bus exists
    if country not in n.buses.index:
        print(f"DEBUG: Bus {country} does not exist yet!")  # Debug print
        n.add("Bus", country)
    
    # Debug print current state
    print(f"DEBUG: Current generators for {country}:")
    print([g for g in n.generators.index if country in g])
    
    for eraa_name, entsoe_name in eraa_to_entsoe.items():
        if eraa_name in future_caps.index:
            gen_name = f"gen_{entsoe_name.lower()}_{country}"
            if gen_name in n.generators.index:
                print(f"DEBUG: Generator {gen_name} already exists")  # Debug print
                continue
                
            if entsoe_name in cap_this.columns:
                print(f"DEBUG: Adding {gen_name}")  # Debug print
                cap_this.loc[:, entsoe_name] = future_caps.loc[eraa_name]
    
    return cap_this

def load_csv_data(country, filepath, col_filter="MapCode"):
    """Load CSV data and filter to the given country."""
    df = pd.read_csv(filepath)
    return df.loc[df[col_filter] == country]

def create_generators(cap, pmin, ramping, country, dem):
    """Create the PyPSA generators dataframe from ENTSO-E capacity data."""
    template = pd.read_csv('data/entsoe_template_generators.csv')
    gen = pypsa_support.generators_from_entsoe(cap, pmin, ramping, template=template).query("p_nom > 0")
    # Set p_min_pu to zero for renewables.
    for carrier in ['onwind', 'offwind', 'solar', 'ror']:
        gen.loc[gen['carrier'] == carrier, 'p_min_pu'] = 0
    # Ensure the 'name' column exists.
    if 'name' not in gen.columns:
        gen['name'] = gen.index
    gen['name'] = "gen_" + gen['name'] + "_" + country
    gen['bus'] = country
    gen.index = gen['name']
    return gen

def add_lol_generator(gen, country):
    """Append a loss-of-load generator to the generator dataframe."""
    # Ensure the original gen has a valid 'name' column.
    if 'name' not in gen.columns:
        gen['name'] = gen.index
    # Create the LoL generator row with a proper index.
    lol = pd.DataFrame({
        'name': f'gen_LoL_{country}',
        'carrier': 'LoL',
        'p_nom': 1e5,
        'efficiency': 1.0,
        'marginal_cost': 1e3,
        'ramp_limit_up': 1.0,
        'ramp_limit_down': 1.0,
        'p_min_pu': 0,
        'bus': country
    }, index=[f'gen_LoL_{country}'])
    
    gen = pd.concat([gen, lol])
    gen.index = gen['name']
    return gen

def get_hydro_inflow(country, start, end, dem):
    """Obtain inflow and minimum storage data for hydropower, converting weekly values to hourly."""
    ext_timeline = pd.date_range(start=start - np.timedelta64(10, 'D'),
                                 end=end + np.timedelta64(10, 'D'),
                                 tz='UTC')
    orig_inflow = metenergy_data.get_inflow_entsoe(
        zone=country, timeline=ext_timeline, MY_API_KEY=ENTSOE_API_KEY
    )
    orig_inflow.loc[orig_inflow['inflow'] < 0, 'inflow'] = 0
    if orig_inflow.index.tz is not None:
        orig_inflow.index = orig_inflow.index.tz_convert(None)
    # Convert weekly values (ffilled) to hourly and divide by 168.
    hourly_inflow = orig_inflow.resample('h').ffill().reindex(dem.index, method='bfill')['inflow'] / 168
    hourly_inflow = pd.DataFrame({f'sto_hydro_{country}': hourly_inflow.values})
    # Minimum storage (from the "storage" column) and lower-bound adjustment.
    min_stor = orig_inflow.resample('h').ffill().reindex(dem.index, method='bfill')['storage']
    min_stor = pd.DataFrame({f'sto_hydro_{country}': min_stor.values})
    min_stor.loc[(min_stor.index % (24 * 7)) != 0] = 0
    return hourly_inflow, min_stor, orig_inflow

def create_storage_units(cap, country):
    """Create the PyPSA storage units dataframe from ENTSO-E capacity data."""
    sto = pypsa_support.stores_from_entsoe(cap)
    sto['name'] = "sto_" + sto['name'] + "_" + country
    sto['bus'] = country
    sto['cyclic_state_of_charge_per_period'] = False
    sto.index = sto['name']
    jrc = pd.read_csv('data/max_hours_jrc.csv')
    if 'PHS' in sto['carrier'].values and country in jrc['country_code'].values:
        sto.loc[sto['carrier'] == 'PHS', 'max_hours'] = jrc.loc[jrc['country_code'] == country, 'h'].values
    return sto

def adjust_storage_units(sto, country, orig_inflow):
    """Adjust hydropower storage units if applicable."""
    storage_key = f'sto_hydro_{country}'
    if storage_key in sto.index and orig_inflow is not None:
        extra = orig_inflow['Hydro Water Reservoir'][0] if 'Hydro Water Reservoir' in orig_inflow.columns else 0
        sto.loc[storage_key, 'state_of_charge_initial'] = orig_inflow['storage'].iloc[0] + extra
        total_storage = orig_inflow['storage'] + orig_inflow.get('Hydro Water Reservoir', 0)
        sto.loc[storage_key, 'max_hours'] = total_storage.max() / sto.loc[storage_key, 'p_nom']
    return sto

def create_pu_series(country, dem, won_cf, sp, ror_cf, wof_cf, gen):
    """Create a dataframe with per-unit maximum generation profiles."""
    data = {
        f'gen_onwind_{country}': won_cf.values,
        f'gen_solar_{country}': sp['sp'].values,
        f'gen_ror_{country}': ror_cf.values
    }
    if 'offwind' in gen['carrier'].values and wof_cf is not None:
        data[f'gen_offwind_{country}'] = wof_cf.values
    return pd.DataFrame(data=data)

def update_pypsa_network(n, country, gen, sto, pu, load, inflow, min_stor):
    """Add the new components (buses, generators, storage, loads) to the PyPSA network."""
    n.madd("Bus", [country])
    # Import generators and storage units using the DataFrame as-is (with valid 'name' and index).
    n.import_components_from_dataframe(gen.set_index('name'), 'Generator')
    n.import_components_from_dataframe(sto.set_index('name'), 'StorageUnit')
    n.import_components_from_dataframe(pd.DataFrame({'bus': country, 'p_set': np.nan}, index=[f'demand_{country}']), 'Load')
    n.import_series_from_dataframe(pu[pu.columns.intersection(gen.index)], 'Generator', 'p_max_pu')
    n.import_series_from_dataframe(load, 'Load', 'p_set')
    
    if inflow is not None and min_stor is not None:
        n.import_series_from_dataframe(inflow, 'StorageUnit', 'inflow')
        n.import_series_from_dataframe(min_stor, 'StorageUnit', 'state_of_charge_set')
        
        selgen = n.storage_units.loc[f'sto_hydro_{country}']
        bus_name = f"{selgen['bus']} {selgen['carrier']}"
        store_name = f"sto_hydro_{country} store {selgen['carrier']}"
        n.add("Bus", bus_name, carrier=selgen["carrier"])
        n.add("Link",
              f'hydro_to_grid_{country}',
              bus0=bus_name,
              bus1=selgen["bus"],
              p_nom=selgen["p_nom"] / selgen["efficiency_dispatch"],
              p_max_pu=1,
              p_min_pu=0,
              marginal_cost=selgen["marginal_cost"] * selgen["efficiency_dispatch"],
              efficiency=selgen["efficiency_dispatch"]
             )
        n.add("Store",
              store_name,
              bus=bus_name,
              e_nom=selgen['p_nom'] * selgen['max_hours'],
              e_nom_extendable=False,
              e_initial=selgen['state_of_charge_initial'],
              e_min_pu=(n.storage_units_t.state_of_charge_set[f'sto_hydro_{country}']) / (selgen['p_nom'] * selgen['max_hours']),
              e_max_pu=1.0
             )
        inflow_max = n.storage_units_t.inflow[f'sto_hydro_{country}'].max()
        inflow_pu = 0 if inflow_max == 0 else n.storage_units_t.inflow[f'sto_hydro_{country}'] / inflow_max
        n.add("Generator",
              f'inflow_hydro_{country}',
              bus=bus_name,
              carrier="inflow",
              p_nom=inflow_max,
              p_max_pu=inflow_pu,
              ramp_limit_up=1,
              ramp_limit_down=1
             )
        n.add("Generator",
              f'LoL_inflow_hydro_{country}',
              bus=bus_name,
              carrier="LoL_inflow",
              p_nom=1e6,
              p_max_pu=1,
              ramp_limit_up=1,
              marginal_cost=2000,
              ramp_limit_down=1
             )
        n.remove("StorageUnit", f'sto_hydro_{country}')

def update_marginal_costs_and_availability(n, gen):
    """Set marginal costs for thermal generators and adjust p_max_pu using monthly availability factors."""
    for gen_name in gen.index:
        carrier = n.generators.at[gen_name, 'carrier']
        if carrier in THERMAL_FUELS:
            mc = calculate_marginal_cost(carrier, fuel_prices_df, n.generators.at[gen_name, 'efficiency'])
            n.add('Generator', gen_name, marginal_cost=mc)
    
    availabilities = pd.read_csv('data/availabilities.csv', index_col=0)
    carrier_to_tech = {
        'nuclear': 'Nuclear',
        'lignite': 'Lignite', 
        'coal': 'Coal',
        'CCGT': 'Gas',
        'oil': 'Oil',
        'biomass': 'Biomass',
        'ror': 'RoR',
        'waste': 'Waste'
    }
    
    snapshot_dates = pd.date_range(start=START, end=END, freq='h', inclusive='left')
    months = snapshot_dates.month
    
    for gen_name in gen.index:
        carrier = n.generators.at[gen_name, 'carrier']
        if carrier in carrier_to_tech:
            tech = carrier_to_tech[carrier]
            if tech in availabilities['technology'].values:
                avail = availabilities.loc[availabilities['technology'] == tech].iloc[0]
                hourly_factors = pd.Series(index=n.snapshots)
                for month in range(2, 14):
                    mask = (months == month)
                    hourly_factors[mask] = avail.iloc[month - 1]
                if gen_name in n.generators_t.p_max_pu:
                    hourly_factors = hourly_factors * n.generators_t.p_max_pu[gen_name]
                if carrier == 'nuclear' and gen_name == 'gen_nuclear_FR':
                    hourly_factors = hourly_factors * 0.91
                n.add('Generator', gen_name, p_max_pu=hourly_factors)

# === MAIN LOOP ===

for country in LIST_COUNTRIES:
    print(f"Adding {country}...")
    
    # 1. Load demand.
    dem = get_demand(country, START, END)
    
    # 2. Wind and solar capacity factors.
    won_cf, wof_cf = get_wind_cf(country, dem)
    sp = get_solar_cf(country, dem, YEAR)
    
    # 3. Generation capacities and adjustments.
    cap_this = get_generation_capacity(country, YEAR)
    cap_this = adjust_capacities(cap_this, country)
    cap_this = replace_other(cap_this, country)
    
    # 4. Load additional CSV data.
    pmin = load_csv_data(country, 'data/mingen-2020_2022.csv')
    ramping = load_csv_data(country, 'data/ramping-2020_2022.csv')
    
    # 5. Create generators.
    gen = create_generators(cap_this, pmin, ramping, country, dem)
    gen = add_lol_generator(gen, country)
    
    # 6. Get hydropower inflow if applicable.
    if 'Hydro Water Reservoir' in cap_this.columns:
        hourly_inflow, min_stor, orig_inflow = get_hydro_inflow(country, START, END, dem)
    else:
        hourly_inflow, min_stor, orig_inflow = None, None, None
    
    # 7. Create storage units.
    sto = create_storage_units(cap_this, country)
    sto = adjust_storage_units(sto, country, orig_inflow)
    
    # 8. Convert run-of-river CF to hourly.
    this_ror = ror.loc[YEAR, country].resample('H').ffill().loc[dem.index]
    
    # 9. Create p_max_pu profiles.
    pu = create_pu_series(country, dem, won_cf, sp, this_ror, wof_cf, gen)
    
    # 10. Create load timeseries.
    load = pd.DataFrame({f'demand_{country}': dem['Actual Load'].values})
    
    # 11. Update the PyPSA network.
    update_pypsa_network(n, country, gen, sto, pu, load, hourly_inflow, min_stor)
    
    # 12. Update thermal marginal costs and availability-based p_max_pu.
    update_marginal_costs_and_availability(n, gen)
    
    
    
# Add links
list_links = (
    pd.read_csv('data/ntc-2020_2022.csv')
    .query('year == 2022')
)
list_links = list_links.loc[ list_links['to'].isin(LIST_COUNTRIES) & list_links['from'].isin(LIST_COUNTRIES)]

n.madd(
    "Link", 
    'link_' + list_links['from'] + '_' + list_links['to'],
    bus0 = list_links['from'].values,
    bus1 = list_links['to'].values,
    p_min_pu = 0,
    p_max_pu = 1,
    p_nom = list_links['max'].values
)

n.generators_t.p_max_pu.fillna(1, inplace=True)
n.generators_t.p_min_pu.fillna(0, inplace=True)
n.stores_t.e_min_pu.fillna(0, inplace=True)
# some demand values can be zero creating infeasibilities
n.loads_t.p_set = n.loads_t.p_set.replace(to_replace=0, method='ffill')

n.consistency_check()


n.optimize(snapshots = range(NSTEPS), solver_name='highs', solver_options={'log_to_console': True})

n.export_to_netcdf('data/model.nc')