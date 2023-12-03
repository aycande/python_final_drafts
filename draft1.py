#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:48:49 2023

@author: denizaycan
"""

import os
import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import wbdata
from country_converter import CountryConverter
import requests
import xml.etree.ElementTree as ET
import pycountry
from pypdf import PdfReader
import re
import statsmodels.api as sm
from linearmodels.panel import PanelOLS





base_path = r'/users/denizaycan/Documents/GitHub/python_final_drafts'

def load_andprep_excel_data(base_path, file_name):
    '''
    This function loads and prepares the excel file by removing extra lines.
    Arguments: base_path (str)
               file_name (str) 
    Returns the df.           
    '''
    full_path = os.path.join(base_path, file_name)
    df = pd.read_excel(full_path)
    return df

ODI_HBS_pro = load_andprep_excel_data(base_path, 'ODI_HBS_projects.xlsx')
ODI_HBS_pro.columns
ODI_HBS_pro['Fund Type'].unique()
ODI_HBS_pro['Sector (OECD)'].unique()

subsector_distribution = ODI_HBS_pro['Sub-Sector'].value_counts().reset_index()
subsector_distribution.columns = ['Sub-Sector', 'Count']
print(subsector_distribution)

sector_distribution = ODI_HBS_pro['Sector (OECD)'].value_counts().reset_index()
sector_distribution.columns = ['Sector (OECD)', 'Count']
print(sector_distribution)

country_distribution = ODI_HBS_pro['Country'].value_counts().reset_index()
country_distribution.columns = ['Country', 'Count']
print(country_distribution)

#Eswatini as Eswatini and "Kingdom of Eswatini", Gambia as Gambia and The Gambia
#Bahamas as The Bahamas and Bahamas
commonwealth_countries = {
    "Africa": [
        "Botswana", "Cameroon", "Gabon", "Gambia", "The Gambia", "Ghana", 
        "Kenya", "Kingdom of Eswatini", "Lesotho", "Malawi", "Mauritius",
        "Mozambique", "Namibia", "Nigeria", "Rwanda", "Seychelles", "Sierra Leone", 
        "South Africa",
        "Eswatini", "Togo", "Uganda", "United Republic of Tanzania", "Zambia"
    ],
    "Asia": [
        "Bangladesh", "Brunei Darussalam", "India", "Malaysia", "Maldives", "Pakistan", 
        "Singapore", "Sri Lanka"
    ],
    "Caribbean and Americas": [
        "Antigua and Barbuda", "Bahamas", "The Bahamas", "Barbados", "Belize",
        "Canada", "Dominica", "Grenada",
        "Guyana", "Jamaica", "Saint Kitts and Nevis", "Saint Lucia", 
        "Saint Vincent and the Grenadines",
        "Trinidad and Tobago"
    ],
    "Europe": [
        "Cyprus", "Malta", "United Kingdom"
    ],
    "Pacific": [
        "Australia", "Fiji", "Kiribati", "Nauru", "New Zealand", 
        "Papua New Guinea", "Samoa",
        "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu"
    ]
}

# Create a list of all Commonwealth countries
all_commonwealth_countries = [country for countries in commonwealth_countries.values() for country in countries]


ODI_cw = ODI_HBS_pro[ODI_HBS_pro['Country'].isin(all_commonwealth_countries) | ODI_HBS_pro['Country'].str.contains('global', case=False)]
total_grant_by_country = ODI_cw.groupby('Country')['Grant'].sum()

plt.figure(figsize=(10, 6))
plt.title('Total Grant Distribution by Country')
plt.xlabel('Country')
plt.ylabel('Total Grant Amount')
total_grant_by_country.plot(kind='bar', color='skyblue')


# Save the plot to the base path
image_path = os.path.join(base_path, 'total_grant_by_country.png')
plt.savefig(image_path)

total_grant_by_year = ODI_cw.groupby('Approved year')['Grant'].sum()
plt.figure(figsize=(10, 6))
plt.title('Total Grant Distribution Over Time')
plt.xlabel('Years')
plt.ylabel('Total Grant Amount')
total_grant_by_year.plot(kind='bar', color='skyblue')

image_path2 = os.path.join(base_path, 'total_grant_by_year.png')
plt.savefig(image_path2)

indicator = 'NY.GDP.PCAP.PP.KD'

cc = CountryConverter()
country_name_to_code = {country: cc.convert(names=country, to='ISO3') for country in all_commonwealth_countries}

# Download data from the World Bank API for all available time periods
dataframes = []
for country_name, country_code in country_name_to_code.items():
    data = wbdata.get_dataframe(indicators={indicator: 'GDP_ppp'}, country=country_code)
    data['Country'] = country_name  # Add a 'Country' column for easy identification
    data.reset_index(inplace=True) 
    dataframes.append(data)

# Concatenate the dataframes into a single dataframe
GDP = pd.concat(dataframes)

#OECD API process
api_url = "https://sdmx.oecd.org/public/rest/data/OECD.ENV.EPI,DSD_CAPMF@DF_CAPMF,1.0/MLT+IND+ZAF+GBR+NZL+CAN+AUS.A.POL_COUNT+POL_STRINGENCY.LEV3_GHG_ACC+LEV3_UNFCCC+LEV3_EVAL_BR+LEV3_BAN_FF_ABROAD+LEV3_BAN_CREDIT+LEV2_INT_GHGREP+LEV2_INT_C_FIN+LEV2_INT_C_COORD+LEV2_CROSS_SEC_CG+LEV2_CROSS_SEC_FFPP+LEV2_CROSS_SEC_RDD+LEV2_CROSS_SEC_GHGTAR+LEV2_SEC_T_NMBI+LEV2_SEC_T_MBI+LEV2_SEC_B_NMBI+LEV2_SEC_B_MBI+LEV2_SEC_I_NMBI+LEV2_SEC_I_MBI+LEV2_SEC_E_NMBI+LEV2_SEC_E_MBI+LEV1_INT+LEV1_CROSS_SEC+LEV1_SEC.0_TO_10+PL?startPeriod=1990&endPeriod=2022&dimensionAtObservation=AllDimensions"
response = requests.get(api_url)

if response.status_code == 200:
    try:
        # Parse XML content
        root = ET.fromstring(response.content)

        # Extract information from XML
        observations = []
        for obs in root.findall(".//{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}Obs"):
            obs_data = {}
            for key in obs.findall(".//{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}ObsKey"):
                for value in key.findall(".//{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}Value"):
                    obs_data[value.attrib['id']] = value.attrib['value']
            obs_value = obs.find(".//{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}ObsValue")
            if obs_value is not None:
                obs_data['ObsValue'] = obs_value.attrib['value']
            observations.append(obs_data)

        # Convert to DataFrame
        OECD = pd.DataFrame(observations)
        print(OECD)

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        print("Response content:", response.text)

else:
    print(f"Error: {response.status_code}")
    
def get_country_code(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_3
    except AttributeError:
        return None

# Example usage:
country_name = "United States"
country_code = get_country_code(country_name)
print(country_code)

ODI_cw['countrycode'] = ODI_cw['Country'].apply(get_country_code)
GDP['countrycode'] = GDP['Country'].apply(get_country_code)



###### pdf parsin part

file = r'https://production-new-commonwealth-files.s3.eu-west-2.amazonaws.com/s3fs-public/2023-08/Climate%20Finance%20Access%20Hub%20Impact%20Brochure%20V2.pdf?VersionId=un05V.PU3ax_p7dH9OH2sJgte8Gt_U9v'

response = requests.get(file)
fname = file.split('/')[-1]
#CONTEXTMANAGEEMENT
with open(os.path.join(base_path, fname), 'wb') as ofile:
    ofile.write(response.content)
#wb as in write brinary

pdf = PdfReader(os.path.join(base_path, fname))
#pdf = instance, pdfreader = class pdf.pages pages = attributeds

print(len(pdf.pages))
print(pdf.get_fields())
print(pdf.is_encrypted)

page_one = pdf.pages[0]
textt = page_one.extract_text()
print(textt)

textt = []
for page in pdf.pages:
    textt.append(page.extract_text())

print(textt[0])

full_textt = '\n'.join(textt)
with open(os.path.join(base_path, 'cw_climatefinance_accesshub.txt'), 'w', encoding='utf-8') as ofile:
    ofile.write(full_textt)


def start_and_end(text, targets):
    '''
    This function finds the location of the relevant lines and remove irrelevant headers.
    Arguments: text(str)
               target(str) : the relevant words to identify start and end of lines
    '''
    lines = text.split('\n')
    line_numbers =[]
    for line_number, line in enumerate(lines, 1):
        for target in targets:
            if target in line:
                line_numbers.append(line_number)
                print(f"Target '{target}' found in line {line_number}: {line}")
                break
    start_line = min(line_numbers) -1
    end_line = max(line_numbers) + 1  # Include the line where the last target is found
    relevant_lines = text.split('\n')[start_line - 1:end_line]
    return relevant_lines


targets = ["Grant of", "Commonwealth Climate Finance Access Hub \ 7"]
relevant_lines_list = start_and_end(full_textt, targets)

if relevant_lines_list:
    print("\nRelevant Lines:")
    for line in relevant_lines_list:
        print(line)
    
for i in range(len(relevant_lines_list)):
    relevant_lines_list[i] = re.sub(r'\bT\s', 'T', relevant_lines_list[i])

Regions = ['Africa', 'Caribbean', 'Pacific']


region_list = []
country_list = []
grant_list = []

current_region = None
current_country = None
current_grant = None

i= 0
# Iterate through lines
while i < len(relevant_lines_list):
    line = relevant_lines_list[i]
    if line in Regions:
        current_region = line
        current_country = None
        i += 1
    # Check if the line is a country
    elif line in all_commonwealth_countries:
        current_country = line
        i += 1
    # Check if the line starts with '• Grant of'
    elif line.startswith(('• Grant of', '• Technical assistance grant of', '• Complimentary finance of')):
        # Reset current_grant for the new '• Grant of' line
        current_grant = line
        i += 1
    else:
        current_grant += ' ' + line if current_grant is not None else line
        i += 1

    # Check if the next line starts with any of the specified prefixes
    next_line_starts_with_prefix = i < len(relevant_lines_list) and relevant_lines_list[i].startswith(
        ('• Grant of', '• Technical assistance grant of', '• Complimentary finance of')
    )
    
    if next_line_starts_with_prefix:
        region_list.append(current_region)
        country_list.append(current_country)
        grant_list.append(current_grant)
        current_grant = None  # Reset current_grant
    elif i == len(relevant_lines_list):
        # Append the last entry if it ends with any of the specified prefixes
        region_list.append(current_region)
        country_list.append(current_country)
        grant_list.append(current_grant)
        
# Create a DataFrame
df_pdf3 = pd.DataFrame({'Region': region_list, 'Country': country_list, 'Grant': grant_list})

df_pdf3.at[0, 'Country'] = 'Africa'

# Uplift the rows under the 'Grant' column by 1
df_pdf3['Grant'] = df_pdf3['Grant'].shift(-1)

# Drop the last row as it will have NaN values after shifting
df_pdf3 = df_pdf3.drop(df_pdf3.index[-1])

# Reset the index
df_pdf3 = df_pdf3.reset_index(drop=True)
df_pdf3['Grant'] = df_pdf3['Grant'].str.replace(r'^.*?\$', '', regex=True)


df_pdf3['Grant_Amount'] = ""
df_pdf3['Grant_Info'] = ""

for index, row in df_pdf3.iterrows():
    grant_value = str(row['Grant'])  # Convert to string to handle non-string values
    
    # Check if 'million' is present in the grant value
    if 'million' in grant_value:
        parts = grant_value.split('million', 1)
        df_pdf3.at[index, 'Grant_Amount'] = float(parts[0].strip()) * 1e6
        df_pdf3.at[index, 'Grant_Info'] =   parts[1].strip()
    else:
        match = re.match(r'([0-9,.]+)\s*(.*)', grant_value)
        if match:
            df_pdf3.at[index, 'Grant_Amount'] = float(match.group(1).replace(',', ''))
            df_pdf3.at[index, 'Grant_Info'] = match.group(2).strip()

# Convert 'Grant_Amount' column to numeric, coercing errors to NaN
df_pdf3['Grant_Amount'] = pd.to_numeric(df_pdf3['Grant_Amount'], errors='coerce')

######## pdf parsing over
OECD['TIME_PERIOD'] = OECD['TIME_PERIOD'].astype(str)

# Convert 'ObsValue' column to numeric
OECD['ObsValue'] = pd.to_numeric(OECD['ObsValue'], errors='coerce')

# Get unique values in the 'CLIM_ACT_POL' column
unique_clim_act_pol = OECD['CLIM_ACT_POL'].unique()

# Set a seaborn color palette for better differentiation
sns.set_palette("husl")

# Loop through unique values in 'CLIM_ACT_POL'
for clim_act_pol_value in unique_clim_act_pol:
    # Filter the DataFrame based on the current 'CLIM_ACT_POL' value and MEASURE == 'POL_STRINGENCY'
    filtered_df = OECD[(OECD['CLIM_ACT_POL'] == clim_act_pol_value) & (OECD['MEASURE'] == 'POL_STRINGENCY')]
    
    # Check if there are any rows to plot
    if not filtered_df.empty:
        # Plotting individual ObsValue for each country over time
        plt.figure(figsize=(12, 8))
        
        # Loop through unique country codes and plot their respective ObsValues
        for country_code in filtered_df['REF_AREA'].unique():
            country_data = filtered_df[filtered_df['REF_AREA'] == country_code]
            plt.plot(country_data['TIME_PERIOD'], country_data['ObsValue'], label=country_code, marker='o', linestyle='-')
        
        # Add labels and title
        plt.xlabel('Year')
        plt.ylabel('Score')
        plt.title(f'Individual Score of Countries on International Climate Finance over Time - {clim_act_pol_value}')
        
        # Show only a subset of years on the x-axis (adjust the step as needed)
        plt.xticks(filtered_df['TIME_PERIOD'].unique()[::2], rotation=45, ha='right')
        
        # Improve legend placement
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Country Code')
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot to the specified base path with a filename based on 'CLIM_ACT_POL' value
        filename = f'{clim_act_pol_value.replace(" ", "_")}_plot.png'
        filepath = os.path.join(base_path, filename)
        plt.savefig(filepath)
        
        # Show the plot for the current 'CLIM_ACT_POL' value
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.show()
    else:
        print(f"No data found for {clim_act_pol_value} and MEASURE == 'POL_STRINGENCY'. Skipping plot.")


def process_OECD_data(OECD):
    """
    Process OECD data with the following steps:
    1. Rename the first column to 'Year' and the second column to 'countrycode'.
    2. Remove the 3rd column.
    3. Limit the 4th column to 'POL_STRINGENCY'.
    4. Remove the 6th column.
    5. Rename the 7th column to 'Score'.

    Parameters:
    - OECD (pd.DataFrame): Input DataFrame containing OECD data.

    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    # Rename the first and second columns
    OECD.rename(columns={'TIME_PERIOD': 'Year', 'REF_AREA': 'countrycode'}, inplace=True)

    # Remove the 3rd column
    OECD.drop(columns=['FREQ'], inplace=True)

    # Limit the 4th column to 'POL_STRINGENCY'
    OECD = OECD[OECD['MEASURE'] == 'POL_STRINGENCY']

    # Remove the 6th column
    OECD.drop(columns=['UNIT_MEASURE'], inplace=True)

    # Rename the 7th column to 'Score'
    OECD.rename(columns={'ObsValue': 'Score'}, inplace=True)

    return OECD

processed_OECD = process_OECD_data(OECD)
processed_OECD.drop(columns=['MEASURE'], inplace=True)

wide_OECD = processed_OECD.pivot(index=['Year', 'countrycode'], columns='CLIM_ACT_POL', values='Score').reset_index()
IND_ZAF_OECD = wide_OECD[wide_OECD['countrycode'].isin(['IND', 'ZAF'])]
IND_ZAF_ODI = ODI_cw[ODI_cw['countrycode'].isin(['IND', 'ZAF'])]
IND_ZAF_GDP = GDP[GDP['countrycode'].isin(['IND', 'ZAF'])]

print(IND_ZAF_ODI.columns)

subset_columns = ['Approved year', 'countrycode']
IND_ZAF_ODI_grouped = IND_ZAF_ODI.groupby(subset_columns).agg({'Amount of Funding Approved (USD millions)': 'sum'}).reset_index()
IND_ZAF_ODI_grouped.rename(columns={'Approved year': 'Year', 'Amount of Funding Approved (USD millions)': 'Funding'}, inplace=True)
print(IND_ZAF_GDP.columns)
IND_ZAF_GDP.rename(columns={'date': 'Year'}, inplace=True)
IND_ZAF_ODI_grouped['Year'] = pd.to_numeric(IND_ZAF_ODI_grouped['Year'], errors='coerce')

# Convert 'Year' column to numeric in 'IND_ZAF_OECD'
IND_ZAF_OECD['Year'] = pd.to_numeric(IND_ZAF_OECD['Year'], errors='coerce')

# Convert 'Year' column to numeric in 'IND_ZAF_GPD'
IND_ZAF_GDP['Year'] = pd.to_numeric(IND_ZAF_GDP['Year'], errors='coerce')

merge_1 = pd.merge(IND_ZAF_ODI_grouped, IND_ZAF_OECD, on=['countrycode', 'Year'], how='inner')
final_merged_df = pd.merge(merge_1, IND_ZAF_GDP, on=['countrycode', 'Year'], how='inner')
print(final_merged_df.columns)

#basic analysis, how does funding impact the emission outcome?
final_merged_df_dummy = pd.get_dummies(final_merged_df, columns=['countrycode', 'Year'], drop_first=True)
y = final_merged_df_dummy['LEV1_CROSS_SEC']
X = final_merged_df_dummy[['Funding'] + [col for col in final_merged_df_dummy.columns if col.startswith('countrycode') or col.startswith('Year')]]
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

final_merged_df.reset_index(inplace=True)
mod = PanelOLS.from_formula('LEV1_CROSS_SEC ~ Funding + GDP_ppp + EntityEffects + TimeEffects', final_merged_df.set_index(['countrycode', 'Year']))
res = mod.fit(cov_type='clustered', cluster_entity=True)
print(res)

mod = PanelOLS.from_formula('Funding ~ GDP_ppp + EntityEffects + TimeEffects', final_merged_df.set_index(['countrycode', 'Year']))
res = mod.fit(cov_type='clustered', cluster_entity=True)
print(res)


#loading the cvsv about the emissions from climate watch

def load_csv_data(base_path, file_name):
    '''
    This function loads and prepares the csv file.
    Arguments: base_path (str)
               file_name (str) 
    Returns the df.           
    '''
    full_path = os.path.join(base_path, file_name)
    df = pd.read_csv(full_path)
    return df

emissions = load_csv_data(base_path, 'ghg-emissions_percapita.csv')
print(emissions.columns)

def process_emissions_data(df):
    """
    Process emissions data with the following steps:
    1. Rename the first column to 'Year' and the second column to 'countrycode'.
    2. Remove the 3rd column.
    3. Limit the 4th column to 'POL_STRINGENCY'.
    4. Remove the 6th column.
    5. Rename the 7th column to 'Score'.

    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    # Rename the first and second columns
    df.rename(columns={'iso': 'countrycode'}, inplace=True)
    # Remove the 3rd column
    df.drop(columns=['Country/Region'], inplace=True)
    df.drop(columns=['unit'], inplace=True)
    return df

processed_emissions = process_emissions_data(emissions)
processed_emissions_long = pd.melt(processed_emissions, id_vars='countrycode', var_name='Year', value_name='Emissions')
processed_emissions_long.reset_index(inplace=True)
subset_columns = ['Approved year', 'countrycode']
ODI_cw_grouped = ODI_cw.groupby(subset_columns).agg({'Amount of Funding Approved (USD millions)': 'sum'}).reset_index()
ODI_cw_grouped.rename(columns={'Approved year': 'Year', 'Amount of Funding Approved (USD millions)': 'Funding'}, inplace=True)
GDP.rename(columns={'date': 'Year'}, inplace=True)
cw_merge_1 = pd.merge(processed_emissions_long, GDP, on=['countrycode', 'Year'], how='inner')
ODI_cw_grouped['Year'] = pd.to_numeric(ODI_cw_grouped['Year'], errors='coerce')
cw_merge_1['Year'] = pd.to_numeric(cw_merge_1['Year'], errors='coerce')
cw_merge_2 = pd.merge(cw_merge_1, ODI_cw_grouped, on=['countrycode', 'Year'], how='inner')
cw_merge_2.drop(columns=['index', 'Country'], inplace=True)
cw_merge_2.columns
cw_merge_2['Emissions'] = pd.to_numeric(cw_merge_2['Emissions'], errors='coerce')
wide_OECD['Year'] = pd.to_numeric(wide_OECD['Year'], errors='coerce')

mod = PanelOLS.from_formula('Emissions ~ GDP_ppp + EntityEffects + TimeEffects', cw_merge_2.set_index(['countrycode', 'Year']))
res = mod.fit(cov_type='clustered', cluster_entity=True)
print(res)

cw_merge_2_dummy = pd.get_dummies(cw_merge_2, columns=['countrycode', 'Year'], drop_first=True)
m = PanelOLS(dependent=cw_merge_2_dummy['Emissions'],
             exog=cw_merge_2_dummy['Funding'],
             entity_effects=True,
             time_effects=True)
results = m.fit(cov_type='clustered', cluster_entity=True)
print(results)