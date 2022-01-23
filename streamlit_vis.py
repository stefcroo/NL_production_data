import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
import requests
import json
from geopy.geocoders import Nominatim
import geopy
from geopy.distance import distance
import math
import numpy as np
import seaborn as sns
import statistics
import base64
import geopandas as gpd
from geopandas import GeoDataFrame
import branca.colormap as cm
import json
import geojson
import matplotlib
import time 

st.set_page_config(page_title='Netherlands Production from 2003-2021',
                page_icon = ':map:',
                layout= 'wide')


def load_production_df():
    df= pd.read_csv('production_molten.csv')
    df['PRODUCTION'] = df['PRODUCTION']*1000
    df = df.fillna(0)
    return df

# Generate list of unique fields in dataframe


def list_unique_fields(df):
     unique_fields = df['FIELD'].unique()
     return unique_fields
# Generate a dictionary of all operators


def ops_dict():
    ops = pd.read_csv('ops.csv')
    ops_dict = dict(zip(ops['FIELD'], ops['OPERATOR']))
    return ops_dict



def production_fields_since2003_dict():
    prod = pd.read_csv('production_fields_since_2003.csv')
    production_dict = dict(zip(prod['FIELD'], prod['PRODUCTION']))
    return production_dict



def map_production_dictionary_to_gdf(_gdf, production_dict):
    gdf['PRODUCTION_SINCE_2003']= gdf['FIELD_NAME'].map(production_dict)
    return gdf



def field_wellnum_dict(df):
    field_wellnums = df.groupby('FIELD')['WELL'].nunique().reset_index()
    field_wellnums_dictionary=  dict(zip(field_wellnums['FIELD'], field_wellnums['WELL']))
    return field_wellnums_dictionary



def _max_width_(prcnt_width:int = 100):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )
_max_width_()


def csv_downloader(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    filename = f"data_download.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Data</a>'
    st.markdown(href, unsafe_allow_html=True)



def load_gdf(fields):
    gdf = gpd.read_file('fields.geojson')
    gdf = gdf.to_crs(epsg = 4326)
    gdf = gdf.loc[gdf['geometry']!=None]
    gdf = gdf.loc[(gdf.LEG_STATUS.str.startswith('Gas'))]
    gdf = gdf[gdf['FIELD_NAME'].isin(fields)]
    gdf['FIELD_DBK'] = gdf['FIELD_DBK'].apply(lambda x: str(x))
    gdf.to_file("test.geojson", driver="GeoJSON")
    return gdf


def map_operator_dictionary(gdf, dictionary):
    gdf['OPERATOR'] = gdf['FIELD_NAME'].map(dictionary)
    return gdf


def pct_share(df):
    years_sum = df.loc[df['YEAR']==2021].groupby(['YEAR', 'FIELD']).agg({'PRODUCTION':'sum'})
    years_sum['pct_share'] = years_sum.groupby(level=0).apply(lambda x:100*x/float(x.sum()))
    years_sum['pct_share'] = years_sum['pct_share'].round(1)
    years_sum = years_sum.reset_index()
    years_sum.to_csv('pct_share.csv')
    dictionary = dict(zip(years_sum['FIELD'], years_sum['pct_share']))
    return dictionary



def pct_to_gdf(gdf, pct_dict):
    gdf['pct_share']= gdf['FIELD_NAME'].map(pct_dict)
    return gdf



def filter_field(df, field):
    df=  df.loc[df['FIELD']==field]
    return df



def filter_well(df, well):
    df=  df.loc[df['WELL']==well]
    return df



def filter_years(df, year_start, year_end):
    df = df.loc[(df['YEAR']>=year_start)& (df['YEAR']<=year_end)]
    return df



def unique_wells(df):
    return df['WELL'].unique()



def unique_fields(df):
    return df['FIELD'].unique()



def map_dictionary_num_wells_gdf(gdf, dictionary):
    gdf['WELL_COUNT'] = gdf['FIELD_NAME'].map(dictionary)
    gdf = gdf.fillna(0)
    return gdf



def field_options(df, unique_fields):
    list1= ['ALL']
    list2 = list1 + list(unique_fields)

    return list2

  


def filter_field_gdf(gdf, field):
    gdf = gdf.loc[gdf['FIELD_NAME']==field].reset_index()
    return gdf



def num_wells(df):
    nums = len(df['WELL'].unique())
    return nums

 


def num_fields(df):
    nums = len(df['FIELD'].unique())
    return nums




def wells_options(df):
    unique_wells = sorted(list(df['WELL'].unique()))
    well_options = ['ALL'] + unique_wells
    return well_options

 


def sorted_unique_years(df):
    unique_years = sorted(list(df['YEAR'].unique()))
    unique_years = [int(x) for x in unique_years]
    return unique_years



def custom_map_scale(gdf):
    custom_scale = (gdf['pct_share'].quantile((0,0.8,0.95,.98,.9963,.9999, 1))).tolist()
    return custom_scale



def calculate_production_since2003(df):
    total = round(df['PRODUCTION'].sum()/1000)    
    return total
 


def lifetime_production_fields(df):
    grouped_field_production_df = df.groupby('FIELD')['PRODUCTION'].sum().round(-3).reset_index()
    dictionary_fields_production_since_2003 = dict(zip(grouped_field_production_df['FIELD'],grouped_field_production_df['PRODUCTION']))
    return dictionary_fields_production_since_2003



def map_field_production_since_2003(gdf, dictionary):
    gdf['PRODUCTION_SINCE_2003']= gdf['FIELD_NAME'].map(dictionary)
    return gdf



def map_pct_share(gdf, pct_dict):
    gdf['pct_share'] = gdf['FIELD_NAME'].map(pct_dict)
    return gdf
 
# 
# 
def add_map_elements(m, custom_scale, gdf):
    gdf = gdf.reset_index(drop=True)
    gdf = gdf.fillna(0)
    geo= gpd.GeoSeries(gdf.set_index('FIELD_DBK')['geometry']).to_json()
    choropleth =folium.Choropleth(geo_data=geo, name='Choropleth',fill_color='Greens', threshold_scale = custom_scale, data=gdf, legend_name='National Production Share (%) in 2021', columns=['FIELD_DBK', 'pct_share'],fill_opacity=0.5, key_on = 'feature.id').add_to(m)

    style_function = lambda x: {'fillColor': '#ffffff', 
                                'color':'#000000', 
                                'fillOpacity': 0.1, 
                                'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.50, 
                                    'weight': 0.1}

    for i in range(len(gdf)):
        row = gdf.loc[[i]]
        feature = folium.features.GeoJson(
        row,
        style_function=style_function,
        control=False,
        zoom_on_click=True,

        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=[
                'FIELD_NAME','OPERATOR', 'PRODUCTION_SINCE_2003', 'pct_share', 'WELL_COUNT'
            ],
            aliases=[
                "Field: ",
                'Operator: ',
                'Production 2003-2021 mln m\u00b3: ',
                'Total Share %: ',
                # ADD PRODUCION FOR last month (Dec2021)
                'Well count:'
            ],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 9px; padding: 4px;"),
            sticky=True
            )
        )
    # for key in choropleth._children:
    #     if key.startswith('color_map'):
    #         del(choropleth._children[key])
        m.add_child(feature)
        m.keep_in_front(feature)
    return m 

st.subheader("The Netherlands' Gas Production Dashboard 2003-2021")
# Load the dataframe with all production data 
df= load_production_df()
# Create a unique list of fields
fields = unique_fields(df)

fields_dropdown = field_options(df,fields)


# Load geodataframe data 
gdf = load_gdf(fields)

# Create dictionary for number of wells
field_wellnums_dictionary = field_wellnum_dict(df)
# Map num wells to geodataframe
gdf = map_dictionary_num_wells_gdf(gdf,field_wellnums_dictionary)

dictionary_fields_production_since_2003 = lifetime_production_fields(df)
gdf = map_field_production_since_2003(gdf,dictionary_fields_production_since_2003)

# Create dictionary of operators-fields from csv
operator_dictionary = ops_dict()
# Map operators to fields in geodataframe. 
gdf = map_operator_dictionary(gdf,operator_dictionary)
production_dict=production_fields_since2003_dict()
gdf = map_production_dictionary_to_gdf(gdf, production_dict)
# Create a dictioanry for % Production Share 
pct_dict = pct_share(df)
# Map percentage Production share for fields to geodataframe

gdf = map_pct_share(gdf, pct_dict)
plt.rcParams["font.family"] = "futura"


field_options = unique_fields(df)


col_l, col_m, col_r = st.columns(3)
with col_l:
    st.markdown("""
    This dashboard contains production data for on- and offshore gas fields from the Dutch National Oil and Gas portal (NLOG). The productiondata spans from 2003 to 2021 and is categorized by field, well, year and month. 
        \n Use the filters customize the figures. By hovering over the map you can find out more information about the fields. 
        \n The data is public, so feel free to download the selected data to your machine. Happy Exploring!
    
    """
    )
    # Create unique list of fields and add 'ALL' option
    field = st.selectbox('Search Field',fields_dropdown)
    if field == 'ALL':
    
        df = filter_field(df, 'Groningen')
    else:      
        gdf = filter_field_gdf(gdf, field)
        df = filter_field(df, field)

    well_options = wells_options(df)    
    well = st.selectbox('Select Well:', well_options)
    if well !='ALL':
        df = filter_well(df,well)

    unique_years = sorted_unique_years(df)
    # Generate years slider
    years_slider = st.slider('Years Slider', min_value=unique_years[0], max_value = unique_years[1], step=1,value=[unique_years[0], unique_years[-1]])

# Filter dataframe by seleted years
df = filter_years(df, years_slider[0], years_slider[1]) 
    
# Create folium map
location =[53.6012, 5.2]
m = folium.Map(location=location, tiles = 'CartoDB positron', control=False,zoom_start=6.8)    # Create custom scale from gdf percentile values

custom_scale = custom_map_scale(gdf)
# Add elements to map
m= add_map_elements(m, custom_scale, gdf)

with col_m:
    # Display map
    folium_static(m, width =500)
# Create geodata from gdf
 


def load_cbs():
    cbs = pd.read_csv('cbs_nat_gas.csv')
    cbs= cbs.rename({'YEARS':'YEAR'}, axis=1)
    return cbs
cbs=load_cbs()
cbs=filter_years(cbs, years_slider[0], years_slider[1])

 
@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def right_plots(cbs,year_start, year_end):
    fig, (ax1,ax2)= plt.subplots(nrows=2, ncols=1, figsize=(10,11))
    ax1.bar(cbs['YEAR'], cbs['PRODUCTION_INDIGENOUS (mln m3)'], alpha = 0.9, label = 'Nat Gas Production',color='#2b8cbe',edgecolor='black')
    ax1.plot(cbs['YEAR'], cbs['IMPORT_NAT_GAS (mln m3)'],label='Gas Imports', color='#e34a33', alpha=1,linewidth=4)
    ax1.set_ylabel('mln m\u00b3', fontsize=20)
    ax1.set_xlabel('Year', fontsize=20)

    ax1.set_xlim([year_start,year_end])

    ax1.set_xticks(np.arange(year_start, year_end, 2))
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_title(f'National Nat Gas Production and Imports {years_slider[0]}-{years_slider[1]}', fontsize=20)
    ax1.legend(loc='upper right',prop={'size': 15})
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid()
    plt.subplots_adjust(left=None, bottom=0.6, right=None, top=None, wspace=None, hspace=None)


    ax2.bar(cbs['YEAR'], cbs['EXPORT_NAT_GAS (mln m3)'], alpha = 0.8, label = 'Export Nat Gas',color='#756bb1',edgecolor='black')
    ax2.plot(cbs['YEAR'], cbs['TOTAL_CONSUMPTION (mln m3)'],label='Total Consumption', color='#000000', alpha=1, linewidth =4)
    ax2.set_ylabel('mln m\u00b3', fontsize=20)
    ax2.set_xlabel('Year', fontsize=20)
    ax2.set_xlim([year_start,year_end])
    ax2.set_ylim([30000,70000])

    ax2.set_xticks(np.arange(year_start,year_end, 2))
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_title(f'National YoY Gas Production / Consumption {years_slider[0]}-{years_slider[1]}', fontsize=20)
    ax2.legend(loc='upper right',prop={'size': 15})
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid()

    plt.tight_layout()
    plt.show()
    return fig

fig = right_plots(cbs,years_slider[0],years_slider[1])
with col_r:
    st.pyplot(fig)

fig_col1, fig_col2, fig_col3= st.columns(3)




def sums_NL_month(field, year_start, year_end):
    sums_NL_m = pd.read_csv('field_year_month_prod.csv')
    if field =='ALL':
        sums_NL_m = sums_NL_m.loc[sums_NL_m['FIELD']>=field]

    sums_NL_m = sums_NL_m.loc[(sums_NL_m['YEAR']>=year_start) & (sums_NL_m['YEAR']<=year_end)]
    return sums_NL_m



def sums_NL_year(year_start,year_end):
    sums_NL_y = pd.read_csv('annual_production.csv')
    sums_NL_y = sums_NL_y.loc[(sums_NL_y['YEAR']>=year_start) & (sums_NL_y['YEAR']<=year_end)] 

    return sums_NL_y




def sums_fields_year(year_start,year_end):
    sums_fields_y = pd.read_csv('sums_fields_y.csv')
    sums_fields_y = sums_fields_y.loc[(sums_fields_y['YEAR']>=year_start) & (sums_fields_y['YEAR']<=year_end)]
    return sums_fields_y



def sums_wells_year(year_start,year_end, field):
    sums_wells_y = pd.read_csv('wells_yby_prod.csv')
    sums_wells_y = sums_wells_y.loc[(sums_wells_y['YEAR']>=year_start) & (sums_wells_y['YEAR']<=year_end)] 
    if field!='ALL':
        sums_wells_y = sums_wells_y.loc[sums_wells_y['FIELD']==field] 
    sums_wells_y  =sums_wells_y.reset_index(drop=True)
    return sums_wells_y

sums_NL_m = sums_NL_month(field, years_slider[0],years_slider[1])
sums_NL_y = sums_NL_year(years_slider[0],years_slider[1])
sums_fields_y = sums_fields_year(years_slider[0],years_slider[1])
sums_wells_y = sums_wells_year(years_slider[0],years_slider[1], field)


# 
def figure3(sums_fields_y, field):
    fig3, ax3= plt.subplots(figsize=(10,6))
    if (field == 'ALL'):
        field='Groningen'
        ax3.set_title(f'{field} Field Production {years_slider[0]}-{years_slider[1]}', fontsize=20)

    elif (field!='ALL'):
        sums_fields_y = filter_field(sums_fields_y, field)
        ax3.set_title(f'{field} Field Production {years_slider[0]}-{years_slider[1]}', fontsize=20)

    ax3.bar(sums_fields_y['YEAR'], sums_fields_y['PRODUCTION'], alpha = 0.8, label = 'Gas Production',color='#2ca25f',edgecolor='black')
    ax3.set_ylabel('mln m\u00b3', fontsize=20)
    ax3.set_xlim([years_slider[0],years_slider[1]])
    ax3.set_xticks(np.arange(years_slider[0],years_slider[1], 2))
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.legend(loc='upper right',prop={'size': 15})
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid()
    plt.tight_layout()
    plt.show()
    return fig3

 

# 
def figure4(sums_wells_y, well):
    fig4, ax4= plt.subplots(figsize=(10,6))
    if (well == 'ALL'):
        
        well = sums_wells_y['WELL'].iloc[0]
        sums_wells_y = filter_well(sums_wells_y, well)
        ax4.set_title(f'{well} Well Annual Production {years_slider[0]}-{years_slider[1]}', fontsize=20)

    elif (well!='ALL'):
        sums_wells_y = filter_well(sums_wells_y, well)
        ax4.set_title(f'{well} Well Annual Production {years_slider[0]}-{years_slider[1]}', fontsize=20)

    ax4.bar(sums_wells_y['YEAR'], sums_wells_y['PRODUCTION'], alpha = 1, label = 'Gas Production',color='#a1d99b',edgecolor='black')
    # ax3.plot(sums_y['YEARS'], cbs['IMPORT_NAT_GAS (mln m3)'],label='Gas Imports', color='r', alpha=.7)
    ax4.set_ylabel('mln m\u00b3', fontsize=20)
    ax4.set_xlim([years_slider[0],years_slider[1]])
    ax4.set_xticks(np.arange(years_slider[0],years_slider[1], 2))
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.legend(loc='upper right',prop={'size': 15})
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid()
    plt.tight_layout()
    plt.show()
    return fig4

 

# 
def figure5(df, field):
    fig5, ax5 = plt.subplots(figsize=(10,6))
    custom_pal = sns.blend_palette(['darkblue', 'lightgreen','yellow','#fc4e2a','brown', 'lightblue'], 12)

    ax5 = sns.barplot(x='MONTH',y='PRODUCTION',data=sums_NL_m,palette = custom_pal, edgecolor='black', order = ['JAN','FEB', 'MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'], alpha=0.5)
    if (field == 'ALL'):
        ax5.set_title(f"Groningen Field Mean Production by Month {years_slider[0]}-{years_slider[1]} (ci=95)", fontsize=20)
    else:
        ax5.set_title(f"{field} Field Mean Production by Month {years_slider[0]}-{years_slider[1]} (ci=95)",fontsize=20)
    ax5.set_ylabel('Production (mln m\u00b3)', fontsize=20)
    ax5.set_xlabel('Month', fontsize=20)
    ax5.tick_params(axis='both', which='major', labelsize=14)
    ax5.grid()
    plt.tight_layout()
    plt.show()
    return fig5

 


def sort_df(df):
    df = df.sort_values(['YEAR', 'FIELD', 'WELL'])
    return df

 


def generate_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    filename = f"data_download.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download csv file</a>'
    return href


fig3 = figure3(sums_fields_y, field)
fig4 = figure4(sums_wells_y, well)
fig5 = figure5(sums_NL_m, field)

with fig_col1:
    # df_sns = df.loc[df['YEAR']==2020]
    st.pyplot(fig3)


with fig_col2:
    st.pyplot(fig5)


with fig_col3:
    st.pyplot(fig4)




checkbox = st.checkbox('Show Data')
if checkbox:
    df = sort_df(df)
    st.dataframe(df)
    href = generate_link(df)
    st.markdown(href, unsafe_allow_html=True)
