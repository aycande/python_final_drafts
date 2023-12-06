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
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shiny.types import ImgData

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

OECD = load_andprep_excel_data(base_path, 'OECD.xlsx')
panel3 = load_andprep_excel_data(base_path, 'panel3.xlsx')
panel3['Year'] = panel3['Year'].astype(str)

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
    'International policies': 'LEV1_INT'
    # Add more mappings as needed
}

column_list = ['GDP_ppp', 'Emissions', 'Empire Total/GtCO2', 'Colonial Total/GtCO2',
               'LEV1_CROSS_SEC', 'LEV1_INT', 'LEV1_SEC', 'LEV2_CROSS_SEC_CG',
               'LEV2_CROSS_SEC_FFPP', 'LEV2_CROSS_SEC_GHGTAR', 'LEV2_CROSS_SEC_RDD',
               'LEV2_INT_C_COORD', 'LEV2_INT_C_FIN', 'LEV2_INT_GHGREP', 'LEV2_SEC_B_MBI',
               'LEV2_SEC_B_NMBI', 'LEV2_SEC_E_MBI', 'LEV2_SEC_E_NMBI', 'LEV2_SEC_I_MBI',
               'LEV2_SEC_I_NMBI', 'LEV2_SEC_T_MBI', 'LEV2_SEC_T_NMBI', 'LEV3_BAN_CREDIT',
               'LEV3_BAN_FF_ABROAD', 'LEV3_EVAL_BR', 'LEV3_GHG_ACC', 'LEV3_UNFCCC']

app_ui = ui.page_fluid(
    # style ----
        ui.panel_title("Final Project by Deniz Aycan: Programming in Python II, Fall 2023"),
        ui.navset_tab(
            ui.nav("Climate Adaptation Scores by OECD",  
            ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.input_selectize("country", "Select Country:", choices = dict(zip(OECD['REF_AREA'].unique(), OECD['REF_AREA'].unique()))),
                        ui.input_selectize("clim_act_pol_names", "Select Policy Action:", choices = list(clim_act_pol_mapping.keys())),
                    ),
                    ui.panel_main(
                        ui.output_plot("climatePlot"),),),),
            ui.nav("Panel Data for Commonwealth: Emissions, GDP and more", 
            ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.input_selectize("country2", "Select Country:", choices = dict(zip(panel3['Country'].unique(), panel3['Country'].unique())),multiple = True),
                        ui.input_selectize("year2", "Select Year:", choices = dict(zip(panel3['Year'].unique(), panel3['Year'].unique()))),
                        ui.input_selectize("x", "Select Variable X:", choices = column_list),
                        ui.input_selectize("y", "Select Variable Y:", choices = column_list),
                    ),
                    ui.panel_main(
                        ui.output_plot("cwPlot"),),),),)
)

def server(input, output, session):
    @reactive.Calc
    def get_data1():
        df = OECD
        df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
        df['ObsValue'] = pd.to_numeric(df['ObsValue'], errors='coerce')
        measure_value = 'POL_STRINGENCY'
        country_code = input.country()
        clim_act_pol_names = input.clim_act_pol_names()
        clim_act_pol_codes = clim_act_pol_mapping.get(clim_act_pol_names)
        if clim_act_pol_codes is not None:
            filtered_df = df[(df['CLIM_ACT_POL'] == clim_act_pol_codes) & (df['MEASURE'] == measure_value) & (df['REF_AREA'] == country_code)]
            print(filtered_df)
        else:
            print(f"Code not found for {clim_act_pol_names}")
        return filtered_df
    @output
    @render.plot
    def climatePlot():
        df = get_data1()
        sns.set()
        plt.figure(figsize=(12, 8))
        df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
        df['ObsValue'] = pd.to_numeric(df['ObsValue'], errors='coerce')
        ax = sns.lineplot(x = df['TIME_PERIOD'], y = df['ObsValue'], hue=df['REF_AREA'], marker='o', linestyle='-')
        ax.set_xlabel('Year')
        ax.set_ylabel('Score')
        ax.set_title(f'Individual Score of Countries on Climate Adaptation over Time: {input.clim_act_pol_names()}')
        ax.set_xticks(df['TIME_PERIOD'].unique()[::2])
        ax.set_xticklabels(df['TIME_PERIOD'].unique()[::2], rotation=45, ha='right')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Country Code')
        plt.grid(True, linestyle='--', alpha=0.7)

        return ax
 
    @reactive.Calc
    def get_data2():
        df = panel3
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        selected_countries = input.country2()
        year = input.year2()
        filtered_df = df[(df['Year'] == year) & (df['Country'].isin(selected_countries))]
        return filtered_df
    @output
    @render.plot
    def cwPlot():
        df = get_data2()
        sns.set()
        plt.figure(figsize=(12, 8))
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        ax = sns.scatterplot(x = df[input.x()], y = df[input.y()], hue=df['Country'], marker='o', linestyle='-')
        ax.set_xlabel('Year')
        ax.set_ylabel('Score')
        ax.set_title(f'Commonwealth countries in {input.x()} and {input.y()} in year {input.year2()}')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Country Code')
        plt.grid(True, linestyle='--', alpha=0.7)

        return ax


app = App(app_ui, server)

df = OECD
df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(str)
df['ObsValue'] = pd.to_numeric(df['ObsValue'], errors='coerce')
measure_value = 'POL_STRINGENCY'
country_code = 'CAN'
clim_act_pol_names = 'Industry - Market-based instruments'
clim_act_pol_codes = clim_act_pol_mapping.get(clim_act_pol_names)
if clim_act_pol_codes is not None:
    # Filter the DataFrame based on the conditions
    filtered_df = df[(df['CLIM_ACT_POL'] == clim_act_pol_codes) & (df['MEASURE'] == measure_value) & (df['REF_AREA'] == country_code)]
    print(filtered_df)
else:
    print(f"Code not found for {clim_act_pol_names}")

print("Available keys in clim_act_pol_mapping:", clim_act_pol_mapping.keys())
print("Selected clim_act_pol_names:", clim_act_pol_names)
