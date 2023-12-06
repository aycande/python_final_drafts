#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:54:55 2023

@author: denizaycan
"""
#Part 3: Plotting
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

plot_merge = load_andprep_excel_data(base_path, 'plot_merge_final.xlsx')
OECD = load_andprep_excel_data(base_path, 'OECD_final.xlsx')
panel3 = load_andprep_excel_data(base_path, 'panel3_final.xlsx')

#non per capita
plt.figure(figsize=(10, 6))
sns.scatterplot(x= 'Colonial Total/GtCO2', y='Pledged (USD million current)', data=plot_merge, hue='Country', palette='viridis')
plt.title('Cross-Country Scatter Plot')
plt.xscale('log')  # Use log scale for x-axis for better visualization
plt.yscale('log')
plt.xlabel('Colonial Total/GtCO2')
plt.ylabel('Pledged (USD million current)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

def get_country_code(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_3
    except AttributeError:
        return None
#per capita graph
filtered_data = plot_merge[(plot_merge['Colonial consumption per yearly capita /tCO2/pc'] > 1000) | (plot_merge['Pledge per Capita (2023)'] > 100)]
filtered_data['countrycode'] = filtered_data['Country'].apply(get_country_code)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Colonial consumption per yearly capita /tCO2/pc', y='Pledge per Capita (2023)', data=filtered_data, hue='Country', palette='viridis')
for i, row in filtered_data.iterrows():
    plt.text(row['Colonial consumption per yearly capita /tCO2/pc'], row['Pledge per Capita (2023)'], row['countrycode'], fontsize=8)
plt.title('Cross-Country Scatter Plot')
plt.xlabel('Colonial consumption per yearly capita /tCO2/pc')
plt.ylabel('Pledge per Capita (2023)')
plt.xscale('log')  # Use log scale for x-axis for better visualization
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plot_path = os.path.join(base_path, 'pledge_emission.png')
plt.savefig(plot_path, dpi=300)
plt.show()


#plot code for shiny

def plot_oecd_variable_over_time(oecd_df, variable_column='CLIM_ACT_POL', country_code=None, clim_act_pol_names=None, measure_value='POL_STRINGENCY'):
    oecd_1 = oecd_df.copy()
    oecd_1['TIME_PERIOD'] = oecd_1['TIME_PERIOD'].astype(str)
    oecd_1['ObsValue'] = pd.to_numeric(oecd_1['ObsValue'], errors='coerce')

    sns.set_palette("husl")

    # Create a mapping of human-readable names to corresponding codes
    clim_act_pol_mapping = {
        'International cooperation': 'LEV2_INT_C_COORD',
        'International public finance': 'LEV2_INT_C_FIN',
        'Banning public finance for fossil fuel infrastructure abroad': 'LEV3_BAN_FF_ABROAD',
        'Evaluation of Biennial Reports and Biennial Update Reports': 'LEV3_EVAL_BR',
        'Climate governance': 'LEV2_CROSS_SEC_CG',
        'Submission of key UNFCCC documents': 'LEV3_UNFCCC',
        'Banning governments\' export credits for new unabated coal power plants': 'LEV3_BAN_CREDIT',
        'GHG emissions data and reporting': 'LEV2_INT_GHGREP',
        'GHG emissions reporting and accounting': 'LEV3_GHG_ACC',
        'Transport - Market-based instruments': 'LEV2_SEC_T_MBI',
        'GHG emission targets': 'LEV2_CROSS_SEC_GHGTAR',
        'Public RD&D expenditure': 'LEV2_CROSS_SEC_RDD',
        'Industry - Non market-based instruments': 'LEV2_SEC_I_NMBI',
        'Electricity - Market-based instruments': 'LEV2_SEC_E_MBI',
        'Cross-sectoral policies': 'LEV1_CROSS_SEC',
        'Industry - Market-based instruments': 'LEV2_SEC_I_MBI',
        'Buildings - Non market-based instruments': 'LEV2_SEC_B_NMBI',
        'Buildings - Market-based instruments': 'LEV2_SEC_B_MBI',
        'Transport - Non market-based instruments': 'LEV2_SEC_T_NMBI',
        'Electricity - Non market-based instruments': 'LEV2_SEC_E_NMBI',
        'Sectoral policies': 'LEV1_SEC',
        'International policies': 'LEV1_INT',
        # Add more mappings as needed
    }

    # If clim_act_pol_names is provided, map them to corresponding codes
    clim_act_pol_codes = [clim_act_pol_mapping[name] for name in clim_act_pol_names] if clim_act_pol_names else None

    for value in clim_act_pol_codes:
        filtered_df = oecd_1[(oecd_1[variable_column] == value) & (oecd_1['MEASURE'] == measure_value)]
        filtered_df = filtered_df[filtered_df['REF_AREA'] == country_code] if country_code else filtered_df

        if not filtered_df.empty:
            full_name = next(key for key, val in clim_act_pol_mapping.items() if val == value)
            plt.figure(figsize=(12, 8))

            for code in filtered_df['REF_AREA'].unique():
                country_data = filtered_df[filtered_df['REF_AREA'] == code]
                plt.plot(country_data['TIME_PERIOD'], country_data['ObsValue'], label=code, marker='o', linestyle='-')

            plt.xlabel('Year')
            plt.ylabel('Score')
            plt.title(f'Individual Score of Countries on Climate Adaptation over Time: {full_name}')
            plt.xticks(filtered_df['TIME_PERIOD'].unique()[::2], rotation=45, ha='right')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Country Code')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            print(f"No data found for {value}, MEASURE == '{measure_value}', and Country Code == '{country_code}'. Skipping plot.")

plot_oecd_variable_over_time(
    OECD,
    variable_column='CLIM_ACT_POL',
    country_code= 'CAN',
    clim_act_pol_names=['International cooperation', 'Transport - Market-based instruments']
)

#secondary shiny 
def plot_variables(data, x_variable, y_variable, year, countries):
    filtered_data = data[(data['Year'] == year) & (data['Country'].isin(countries))]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=filtered_data, x=x_variable, y=y_variable, hue='Country', palette='Set1', s=150, edgecolor='w', linewidth=0.5)
    
    plt.title(f'{y_variable} vs {x_variable} ({year})', fontsize=16)
    plt.xlabel(x_variable, fontsize=14)
    plt.ylabel(y_variable, fontsize=14)

    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Country', fontsize=12)
    
    # Add a nice background color
    plt.gca().set_facecolor('#f8f9fa')

    # Customize tick label fonts
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a legend title
    legend = plt.legend(title='Country', fontsize=10)
    legend.get_title().set_fontsize(12)

    # Add a border around the legend
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor('black')

    # Remove spines
    sns.despine()

    plt.show()


# Example usage:
x_variable = 'GDP_ppp'
y_variable = 'Emissions'
year = 2020
selected_countries = ['Tanzania', 'South Africa', 'United Kingdom', 'India']

plot_variables(panel3, x_variable, y_variable, year, selected_countries)


