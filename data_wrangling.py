#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:44:03 2023

@author: denizaycan
"""
#Part 1: Data Wrangling
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
colonial_for_plot = load_csv_data(base_path, 'full_summary_table_2023.csv')

def download_wb_data(indicator):
    data = wbdata.get_dataframe(indicators={indicator: 'GDP_ppp'}, country='all')
    data.reset_index(inplace=True) 
    return data

GDP_PPP_1 = download_wb_data('NY.GDP.PCAP.PP.KD')

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

def process_OECD_data(df):
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
    - pd.DataFrame: Processed DataFrame."""
    df1 = df.copy()
    df1 = df1.rename(columns={'TIME_PERIOD': 'Year', 'REF_AREA': 'countrycode'})
    df1 = df1.drop(columns=['FREQ'])
    df1 = df1[df1['MEASURE'] == 'POL_STRINGENCY']
    df1 = df1.drop(columns=['MEASURE'])
    df1 = df1.drop(columns=['UNIT_MEASURE'])
    df1 = df1.rename(columns={'ObsValue': 'Score'})
    
    return df1

processed_OECD = process_OECD_data(OECD)
wide_OECD = processed_OECD.pivot(index=['Year', 'countrycode'], columns='CLIM_ACT_POL', values='Score').reset_index()

#I will do the merges based on countrycodes, so I am adding countrycodes to my dfs
def get_country_code(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_3
    except AttributeError:
        return None

#Country dictionary
commonwealth_countries = {
    "Africa": [
        "Botswana", "Cameroon", "Gabon", "Gambia", "The Gambia", "Gambia, The", "Ghana", 
        "Kenya", "Kingdom of Eswatini", "Lesotho", "Malawi", "Mauritius",
        "Mozambique", "Namibia", "Nigeria", "Rwanda", "Seychelles", "Sierra Leone", 
        "South Africa",
        "Eswatini", "Togo", "Uganda", "Tanzania", "United Republic of Tanzania", "Tanzania", "Zambia"
    ],
    "Asia": [
        "Bangladesh", "Brunei Darussalam", "India", "Malaysia", "Maldives", "Pakistan", 
        "Singapore", "Sri Lanka"
    ],
    "Caribbean and Americas": [
        "Antigua and Barbuda", "Bahamas", "The Bahamas", "Bahamas, The", "Barbados", "Belize",
        "Canada", "Dominica", "Grenada",
        "Guyana", "Jamaica", "Saint Kitts and Nevis", "St. Kitts and Nevis", "Saint Lucia", "St. Lucia",
        "Saint Vincent and the Grenadines", "St. Vincent and the Grenadines", 
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


ODI_projects_cw = ODI_projects[ODI_projects['Country'].isin(all_commonwealth_countries) | ODI_projects['Country'].str.contains('global', case=False)]
ODI_projects_cw['countrycode'] = ODI_projects_cw['Country'].apply(get_country_code)

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

#Data preparation for plots:
def process_col_for_plot_emissions(df):
    """
    Process emissions data with the following steps:
    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    df1 = df.copy()
    df1.rename(columns={'country': 'Country'}, inplace=True)
    df2= df1[['Country','Population (2023), millions', 'Colonial Total/GtCO2', 'Colonial consumption per yearly capita /tCO2/pc']]
    df2['Country'] = df2['Country'].replace('USA', 'United States')
    df2['Country'] = df2['Country'].replace('Russia', 'Russian Federation')

    return df2

processed_colonial_for_plot = process_col_for_plot_emissions(colonial_for_plot)

def process_ODI_pledges(df):
    """
    Process emissions data with the following steps:
    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    df1 = df.copy()
    df1.rename(columns={'country': 'Country'}, inplace=True)  
    # Select relevant columns
    df2 = df1[['Country', 'Pledged (USD million current)']]  
    # Group by country and calculate the sum of pledges
    grouped_df = df2.groupby('Country').agg({'Pledged (USD million current)': 'sum'}).reset_index()  
    # Calculate the percentage of each country's pledge to the total sum of pledges
    grouped_df['Pledge Percentage (Non per capita)'] = (grouped_df['Pledged (USD million current)'] / grouped_df['Pledged (USD million current)'].sum()) * 100
    return grouped_df

processed_ODI_pledges = process_ODI_pledges(ODI_pledges)

def merge_and_plot_data(df1, df2):
    """
    Merge two dataframes, calculate colonial emissions percentage, and pledge per capita.

    Parameters:
    - df1 (pd.DataFrame): The first dataframe containing processed ODI pledges data.
    - df2 (pd.DataFrame): The second dataframe containing processed colonial data for plotting.

    Returns:
    - pd.DataFrame: Merged dataframe with additional calculated columns.
    """
    plot_merge = pd.merge(df1, df2, on=['Country'], how='left')
    plot_merge = plot_merge.dropna()
    total_colonial_emissions = plot_merge['Colonial Total/GtCO2'].sum()
    plot_merge['Colonial Percentage/GtCO2'] = (plot_merge['Colonial Total/GtCO2'] / total_colonial_emissions) * 100
    plot_merge['Pledge per Capita (2023)'] = (plot_merge['Pledged (USD million current)'] / plot_merge['Population (2023), millions'])
    
    return plot_merge

plot_merge = merge_and_plot_data(processed_ODI_pledges, processed_colonial_for_plot)

#Data preparation for panel regressions and shiny plots
def process_gdp_data(df):
    df['countrycode'] = df['country'].apply(get_country_code)
    country_code_mapping = {
        'British Virgin Islands': 'VGB',
        'Channel Islands': 'CHI',
        'St. Kitts and Nevis': 'KNA',
        'St. Lucia': 'LCA',
        'St. Vincent and the Grenadines': 'VCT',
        'Tanzania': 'TZA',
        'Gambia, The': 'GMB',
        'Bahamas, The': 'BHS'
    }
    df['countrycode'] = df.apply(lambda row: country_code_mapping.get(row['country'], row['countrycode']), axis=1)
    df_cw = df[df['country'].isin(all_commonwealth_countries)]
    df_cw.rename(columns={'date': 'Year'}, inplace=True)

    
    return df_cw

GDP_PPP_2 = process_gdp_data(GDP_PPP_1)

def process_emissions_data(df):
    """
    Process emissions data with the following steps:
    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    df1 = df.copy()
    df1.rename(columns={'iso': 'countrycode'}, inplace=True)
    df_cw = df1[df1['Country/Region'].isin(all_commonwealth_countries)]
    df_cw.drop(columns=['Country/Region'], inplace=True)
    df_cw.drop(columns=['unit'], inplace=True)
    df2 = pd.melt(df_cw, id_vars='countrycode', var_name='Year', value_name='Emissions')
    df2['Emissions'] = pd.to_numeric(df2['Emissions'], errors='coerce')

    return df2

processed_emissions = process_emissions_data(emissions)

def process_col_emissions_data(df):
    """
    Process emissions data with the following steps:
    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    df1 = df.copy()
    df1.rename(columns={'iso_code': 'countrycode'}, inplace=True)
    df1.rename(columns={'country': 'Country'}, inplace=True)
    df1.rename(columns={'year': 'Year'}, inplace=True)
    df1.drop(columns=['Unnamed: 0', 'region',
       'Fossil CO2/GtCO2',
       'Land CO2/GtCO2', 'Territorial Total/GtCO2', 'Fossil Cons CO2/GtCO2',
       'Consumption Total/GtCO2', 'Emp Fossil Total/GtCO2',
       'Col Fossil Total/GtCO2', 'Emp Land Total/GtCO2',
       'Col Land Total/GtCO2', 'Emp Fossil Cons Total/GtCO2',
       'Col Fossil Cons Total/GtCO2', 'Emp Cons Total/GtCO2', 'Col Cons Total/GtCO2'], inplace=True)
    df_cw = df1[df1['Country'].isin(all_commonwealth_countries)]
    
    return df_cw

processed_colonial_emissions = process_col_emissions_data(colonial_emissions)

panel1 = pd.merge(GDP_PPP_2, processed_emissions, on=['countrycode', 'Year'], how='right')
panel1['Year'] = pd.to_numeric(panel1['Year'], errors='coerce')
processed_colonial_emissions['Year'] = pd.to_numeric(processed_colonial_emissions['Year'], errors='coerce')
panel2 = pd.merge(panel1, processed_colonial_emissions, on=['countrycode', 'Year'], how='left')
wide_OECD['Year'] = pd.to_numeric(wide_OECD['Year'], errors='coerce')
panel3 = pd.merge(panel2, wide_OECD, on=['countrycode', 'Year'], how='left')

#Saving the datasets as excels and lists:
    #Datasets for text processing
file_path = os.path.join(base_path, 'all_commonwealth_countries.txt')
with open(file_path, 'w') as f:
    for item in all_commonwealth_countries:
        f.write(str(item) + '\n')
ODI_projects_excel_file_path = os.path.join(base_path, 'ODI_projects_final.xlsx')
ODI_projects.to_excel(ODI_projects_excel_file_path, index=False)
ODI_projects_cw_excel_file_path = os.path.join(base_path, 'ODI_projects_cw_final.xlsx')
ODI_projects_cw.to_excel(ODI_projects_cw_excel_file_path, index=False)
UK_ICF_long_excel_file_path = os.path.join(base_path, 'UK_ICF_long_final.xlsx')
UK_ICF_long.to_excel(UK_ICF_long_excel_file_path, index=False)

    #Datasets for plotting & regressions
OECD_excel_file_path = os.path.join(base_path, 'OECD_final.xlsx')
OECD.to_excel(OECD_excel_file_path, index=False)
PANEL_excel_file_path = os.path.join(base_path, 'panel3_final.xlsx')
panel3.to_excel(PANEL_excel_file_path, index=False)
plot_merge_excel_file_path = os.path.join(base_path, 'plot_merge_final.xlsx')
plot_merge.to_excel(plot_merge_excel_file_path, index=False)



