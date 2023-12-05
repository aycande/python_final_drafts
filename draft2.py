#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:50:37 2023

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
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from wordcloud import WordCloud
import nltk

base_path = r'/users/denizaycan/Documents/GitHub/python_final_drafts'

#Part 1: downloading datas
def parse_excel_tabs(base_path, file_name):
    """
    Parse each tab of an Excel file as separate DataFrames and print the results.

    Parameters:
    - base_path (str): Base path where the Excel file is located.
    - file_name (str): Name of the Excel file.

    Returns:
    - dict: A dictionary where keys are sheet names and values are corresponding DataFrames.
    """
    # Construct the full file path
    file_path = os.path.join(base_path, file_name)
    all_dfs = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in all_dfs.items():
        print(f"Sheet Name: {sheet_name}")
        print(df)
        print("\n")
    return all_dfs

result = parse_excel_tabs(base_path, 'CFU-Website-MASTER-February-2023.xlsx')
if result:
    ODI_pledges = result['Pledges']
    ODI_projects = result['Projects']

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

UK_ICF = load_andprep_excel_data(base_path, 'UK ICF data 2011-2023_read only.xlsx')

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
colonial_emissions = load_csv_data(base_path, 'full_year_emissions_summary_table_1850_2023.csv')

def download_wb_data(indicator):
    data = wbdata.get_dataframe(indicators={indicator: 'GDP_ppp'}, country='all')
    data.reset_index(inplace=True) 
    return data

wb = download_wb_data('NY.GDP.PCAP.PP.KD')

def get_oecd_data(api_url):
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
            return OECD

        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            print("Response content:", response.text)
            return None

    else:
        print(f"Error: {response.status_code}")
        return None

api_url = "https://sdmx.oecd.org/public/rest/data/OECD.ENV.EPI,DSD_CAPMF@DF_CAPMF,1.0/MLT+IND+ZAF+GBR+NZL+CAN+AUS.A.POL_COUNT+POL_STRINGENCY.LEV3_GHG_ACC+LEV3_UNFCCC+LEV3_EVAL_BR+LEV3_BAN_FF_ABROAD+LEV3_BAN_CREDIT+LEV2_INT_GHGREP+LEV2_INT_C_FIN+LEV2_INT_C_COORD+LEV2_CROSS_SEC_CG+LEV2_CROSS_SEC_FFPP+LEV2_CROSS_SEC_RDD+LEV2_CROSS_SEC_GHGTAR+LEV2_SEC_T_NMBI+LEV2_SEC_T_MBI+LEV2_SEC_B_NMBI+LEV2_SEC_B_MBI+LEV2_SEC_I_NMBI+LEV2_SEC_I_MBI+LEV2_SEC_E_NMBI+LEV2_SEC_E_MBI+LEV1_INT+LEV1_CROSS_SEC+LEV1_SEC.0_TO_10+PL?startPeriod=1990&endPeriod=2022&dimensionAtObservation=AllDimensions"
OECD = get_oecd_data(api_url)

#Country dictionary
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

#Part 2: PDF Parsing

def download_and_process_pdf(output_path, file_url):
    response = requests.get(file_url)
    if response.status_code == 200:
        try:
            file_name = file_url.split('/')[-1]
            with open(os.path.join(output_path, file_name), 'wb') as pdf_file:
                pdf_file.write(response.content)
            pdf = PdfReader(os.path.join(output_path, file_name))
            print(f"Number of pages: {len(pdf.pages)}")
            print(f"Fields: {pdf.get_fields()}")
            print(f"Is encrypted: {pdf.is_encrypted}")
            page_texts = [page.extract_text() for page in pdf.pages]
            full_text = '\n'.join(page_texts)
            with open(os.path.join(output_path, 'cw_climatefinance_accesshub.txt'), 'w', encoding='utf-8') as text_file:
                text_file.write(full_text)

        except Exception as e:
            print(f"Error processing PDF: {e}")
    else:
        print(f"Error downloading PDF: {response.status_code}")
    return full_text

file_url = 'https://production-new-commonwealth-files.s3.eu-west-2.amazonaws.com/s3fs-public/2023-08/Climate%20Finance%20Access%20Hub%20Impact%20Brochure%20V2.pdf?VersionId=un05V.PU3ax_p7dH9OH2sJgte8Gt_U9v'

textfile = download_and_process_pdf(base_path, file_url)

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
relevant_lines_list = start_and_end(textfile, targets)

for i in range(len(relevant_lines_list)):
    relevant_lines_list[i] = re.sub(r'\bT\s', 'T', relevant_lines_list[i])

def process_lines(relevant_lines_list, all_commonwealth_countries):
    Regions = ['Africa', 'Caribbean', 'Pacific']
    
    region_list = []
    country_list = []
    grant_list = []

    current_region = None
    current_country = None
    current_grant = None

    i = 0
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
    return df_pdf3

commonwealth_df1 = process_lines(relevant_lines_list, all_commonwealth_countries)

def process_grant_data(df):
    # Set the value in the first row of the 'Country' column to 'Africa'
    df.at[0, 'Country'] = 'Africa'
    df['Grant'] = df['Grant'].shift(-1)
    df = df.drop(df.index[-1])
    df = df.reset_index(drop=True)

    # Remove the dollar signs and spaces in the 'Grant' column
    df['Grant'] = df['Grant'].str.replace(r'^.*?\$', '', regex=True)
    df['Grant_Amount'] = ""
    df['Grant_Info'] = ""

    for index, row in df.iterrows():
        grant_value = str(row['Grant'])  # Convert to string to handle non-string values

        if 'million' in grant_value:
            parts = grant_value.split('million', 1)
            df.at[index, 'Grant_Amount'] = float(parts[0].strip()) * 1e6
            df.at[index, 'Grant_Info'] = parts[1].strip()
        else:
            match = re.match(r'([0-9,.]+)\s*(.*)', grant_value)
            if match:
                df.at[index, 'Grant_Amount'] = float(match.group(1).replace(',', ''))
                df.at[index, 'Grant_Info'] = match.group(2).strip()

    df['Grant_Amount'] = pd.to_numeric(df['Grant_Amount'], errors='coerce')

    return df

commonwealth_final = process_grant_data(commonwealth_df1)

#Data preparation for word clouds
ODI_projects.rename(columns={'Approved year': 'Year', 'Amount of Funding Approved (USD millions)': 'Grant', 'Summary ': 'Summary'}, inplace=True)

def process_ICF_data(df):
    df1 = df.copy()
    df1.drop(columns=['Unnamed: 0', 'URL','Dept', 'Start','End', 'Total ICF spend £000'], inplace=True)
    column_mapping = {
        '2011/12 ICF spend £000': 2011,
        '2012/13 ICF spend £000': 2012,
        '2013/14 ICF spend £000': 2013,
        '2014/15 ICF spend £000': 2014,
        '2015/16 ICF spend £000': 2015,
        '2016/17 ICF spend £000': 2016,
        '2017/2018 ICF spend £000': 2017,
        '2018/2019 ICF spend £000': 2018,
        '2019/2020 ICF spend £000': 2019,
        '2020/2021 ICF spend £000': 2020,
        '2021/2022 ICF spend £000': 2021,
        '2022/2023 ICF spend £000': 2022
    }
    df1.rename(columns=column_mapping, inplace=True)
    columns_to_stay = ['Project title', 'Country/region', 'Description']
    df_long = pd.melt(df1, id_vars=columns_to_stay, var_name='Year', value_name= 'Grant')
    df_long = df_long.dropna(subset=['Grant'])
    df_long.rename(columns={'Country/region' : 'Country', 'Description': 'Summary'}, inplace=True)

    return df_long

UK_ICF_long = process_ICF_data(UK_ICF)
commonwealth_final.rename(columns={'Grant': 'Full_Info', 'Grant_Amount': 'Grant', 'Grant_Info': 'Summary'}, inplace=True)

#Part 3: Word Clouds
def generate_wordcloud_from_columns(df, summary_column, year_column=None, filter_year=None):
    if year_column is not None and filter_year is not None:
        # Filter data by the specified year
        df_filtered = df[df[year_column] == filter_year]
        # Drop NaN values under the 'summary' column
        df_filtered = df_filtered.dropna(subset=[summary_column])
        # Concatenate both columns if year_column is provided
        text = ' '.join(df_filtered[summary_column].astype(str))
    else:
        # Drop NaN values under the 'summary' column
        df_no_na = df.dropna(subset=[summary_column])
        # Use only the summary_column if no year_column is provided or filter_year is not specified
        text = ' '.join(df_no_na[summary_column].astype(str))
    
    wc = WordCloud(
    background_color='white',  # Background color of the word cloud
    max_words=200,              # Maximum number of words to include in the word cloud
    colormap='viridis',         # Colormap for coloring the words
    contour_width=1,            # Width of the contour lines
    contour_color='black',      # Color of the contour lines
    width=800, height=400,      # Width and height of the word cloud figure
    random_state=42             # Seed for reproducibility
)

    wordcloud = wc.generate(text)
    
    # Display the Word Cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.show()

# Example Usage:
generate_wordcloud_from_columns(ODI_projects, 'Summary', 'Year', filter_year=2015) 
generate_wordcloud_from_columns(ODI_projects, 'Summary', 'Year', filter_year=2022) 

generate_wordcloud_from_columns(UK_ICF_long, 'Summary')
