import pandas as pd
def read_eraa_capacities(year=2025):
    df = pd.read_excel('data/ERAA2023 PEMMDB Generation.xlsx', sheet_name=f'TY{year}', skiprows=1, index_col=1)
    return df.iloc[:23, 1:]


def read_eraa_capacities_country(country):
    a = []
    years = [2025, 2028, 2030, 2033]
    for year in years:
        df_year = read_eraa_capacities(year)
        # Find all columns that start with the country code
        country_cols = [col for col in df_year.columns if col.startswith(country.upper())]
        # Sum all zones for the country
        if country_cols:
            country_data = df_year[country_cols].sum(axis=1)
        else:
            raise ValueError(f"No data found for country {country}")
        a.append(country_data)
    
    df = pd.concat(a, axis=1).T
    df.index = pd.to_datetime([f"{year}-01-01" for year in years])
    # Resample to monthly frequency and interpolate between years
    df = df.astype(float)
    #df = df.resample('A').interpolate(method='linear')
    return df.resample('YS').mean().interpolate(method='linear').round(0)


def read_eraa_capacities_country_year(country, year):
    df = read_eraa_capacities_country(country)
    # Convert year to timestamp if it's a string
    if isinstance(year, str):
        year = pd.Timestamp(year)
    # Get capacities for year and year+1
    year_cap = df.loc[year]
    year_plus_1_cap = df.loc[year + pd.DateOffset(years=1)]
    # Return midpoint between the two years
    return (year_cap + year_plus_1_cap) / 2






