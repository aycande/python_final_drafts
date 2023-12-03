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
    data = wbdata.get_dataframe(indicators={indicator: 'GDP_ppp'}, country=country_code, convert_date=False)
    data['Country'] = country_name  # Add a 'Country' column for easy identification
    dataframes.append(data)

# Concatenate the dataframes into a single dataframe
GDP = pd.concat(dataframes)


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

# Iterate through lines
for line in relevant_lines_list:
    # Check if the line is a region
    if line in Regions:
        current_region = line
        current_country = None
    # Check if the line is a country
    elif line in all_commonwealth_countries:
        current_country = line
    # Check if the line starts with '• Grant of'
    elif line.startswith('• Grant of'):
        # Append data to lists
        region_list.append(current_region)
        country_list.append(current_country)
        grant_list.append(current_grant)  # Append the previous '• Grant of' line

        # Reset current_grant for the new '• Grant of' line
        current_grant = line
    else:
        # If the line doesn't match any condition, merge with the previous '• Grant of' line
        current_grant += ' ' + line if current_grant is not None else line

# Create a DataFrame
df_pdf = pd.DataFrame({'Region': region_list, 'Country': country_list, 'Grant': grant_list})

# Display the DataFrame
print(df_pdf)


region_list = []
country_list = []
grant_list = []

current_region = None
current_country = None
current_grant = None

# Iterate through lines
for line in relevant_lines_list:
    # Check if the line is a region
    if line in Regions:
        current_region = line
        current_country = None
    # Check if the line is a country
    elif line in all_commonwealth_countries:
        current_country = line
    # Check if the line starts with '• Grant of'
    elif line.startswith('• Grant of'):
        # Reset current_grant for the new '• Grant of' line
        current_grant = line
        # Append data to lists
# Append the previous '• Grant of' line
    else:
        # If the line doesn't match any condition, merge with the previous '• Grant of' line
        current_grant += ' ' + line if current_grant is not None else line
        region_list.append(current_region)
        country_list.append(current_country)
        grant_list.append(current_grant)  

# Create a DataFrame
df_pdf2 = pd.DataFrame({'Region': region_list, 'Country': country_list, 'Grant': grant_list})


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
    elif line.startswith('• Grant of'):
        # Reset current_grant for the new '• Grant of' line
        current_grant = line
        i += 1
    else:
        current_grant += ' ' + line if current_grant is not None else line
        i += 1

    if i < len(relevant_lines_list) and relevant_lines_list[i].startswith('• Grant of'):
        region_list.append(current_region)
        country_list.append(current_country)
        grant_list.append(current_grant)
        current_grant = None  # Reset current_grant
    elif i == len(relevant_lines_list):
        # Append the last entry if it ends with '• Grant of'
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
df_pdf3['Grant'] = df_pdf3['Grant'].str.replace('• Grant of \$', '')


df_pdf3['Grant_Amount'] = ""
df_pdf3['Grant_Info'] = ""

for index, row in df_pdf3.iterrows():
    grant_value = str(row['Grant'])  # Convert to string to handle non-string values
    
    # Check if 'million' is present in the grant value
    if 'million' in grant_value:
        parts = grant_value.split('million', 1)
        df_pdf3.at[index, 'Grant_Amount'] = parts[0].strip() 
        df_pdf3.at[index, 'Grant_Info'] =   parts[1].strip()
    else:
        Grant_Amount = ''.join(c for c in grant_value if c.isdigit() or c == '.')
        Grant_Info = grant_value[len(Grant_Amount):].strip()
        df_pdf3.at[index, 'Grant_Amount'] = Grant_Amount
        df_pdf3.at[index, 'Grant_Info'] = Grant_Info
