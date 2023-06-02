import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime
#%matplotlib inline

# Read the Data
df = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/DataFusion-Auvergne-csv2.csv")

# Convert multiple columns to numeric dtype
numeric_columns = ['Consumption', 'Thermal', 'Nuclear', 'Wind', 'Solar', 'Hydro', 'Pumping', 'Bioenergy']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Fill NaN values with interpolation
df.interpolate(inplace=True)

# Convert Date column to datetime data type
df['Date'] = pd.to_datetime(df['Date'])
# Convert "Hour" column to datetime
df['Hour'] = pd.to_datetime(df['Hour'], format='%H:%M').dt.strftime('%H:%M')

df.dropna(subset=['Hour'], how='any', inplace=True)

# Convert the 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Extract the hour from the 'Hour' column and convert to numeric type
df['Hour'] = pd.to_numeric(df['Hour'].str.split(':').str[0])

# Group by 'Region', 'Date', and 'Hour', and calculate the mean for numeric columns
df_grouped = df.groupby(['Date','Hour']).mean(numeric_only=True).reset_index()

# Convert the 'Date' column to datetime type
df_grouped['Date'] = pd.to_datetime(df_grouped['Date'])
# group the data by Date column and calculate the sum of numeric columns
df_grouped2 = df.groupby('Date').sum(numeric_only=True).reset_index()
# adding Electricity Production column
df_grouped2['Production'] = df_grouped2[['Thermal', 'Nuclear', 'Wind', 'Solar', 'Hydro', 'Pumping', 'Bioenergy']].sum(axis=1)

df_grouped2.tail(20)
#------------------------------------------------------------------------------Time series production and consumption
import plotly.graph_objects as go

# Convert the date column to a datetime object
df_grouped2['Date'] = pd.to_datetime(df_grouped2['Date'])

# Create the figure and traces
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_grouped2['Date'], y=df_grouped2['Consumption'], name='Consumption',
                         line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=df_grouped2['Date'], y=df_grouped2['Production'], name='Production',
                         line=dict(color='#0343DF', width=2),  yaxis='y2'))

# Set the axis titles
fig.update_layout(
    xaxis=dict(title='Date'),
    yaxis=dict(title='Consumption', titlefont=dict(color='red')),
    yaxis2=dict(title='Production', titlefont=dict(color='blue'), overlaying='y', side='right'),
    
    # Set the background color to white
    plot_bgcolor='white'
)

# Add hover information
fig.update_traces(hovertemplate='%{y:.2f}')

# Show the figure
fig.show()

#-------------------------------------------------------------------------Boxplot
df_boxplot= df_grouped2.copy()

import plotly.subplots as sp
import plotly.express as px
import pandas as pd


df_boxplot = df_boxplot.set_index('Date')  # set 'Date' column as index
df_boxplot.index = pd.to_datetime(df_boxplot.index)  # convert to datetime data type

print(df_boxplot.index.isnull().sum()) # prints the number of missing values
print(df_boxplot.index.duplicated().sum()) # prints the number of duplicated values


# Create four dataframes for the four seasons
df_spring = df_boxplot[(df_boxplot.index >= '2022-03-01') & (df_boxplot.index <= '2022-05-31')]
df_summer = df_boxplot[(df_boxplot.index >= '2022-06-01') & (df_boxplot.index <= '2022-08-31')]
df_autumn = df_boxplot[(df_boxplot.index >= '2022-09-01') & (df_boxplot.index <= '2022-11-30')]
df_winter = df_boxplot[(df_boxplot.index >= '2022-12-01') & (df_boxplot.index <= '2023-02-28')]

# Create a list of day names to use for X axis labels
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create subplots for each season
fig_boxplot = sp.make_subplots(rows=2, cols=2, subplot_titles=('Spring Consumption', 'Summer Consumption',
                                                       'Autumn Consumption', 'Winter Consumption'),
                       shared_xaxes=True, vertical_spacing=0.1)

# Add each boxplot to the subplots
fig_boxplot.add_trace(px.box(df_spring, x=df_spring.index.day_name(), y='Consumption', category_orders={'x': day_names}).data[0].update(marker=dict(color='blue')),row=1, col=1)

fig_boxplot.add_trace(px.box(df_summer, x=df_summer.index.day_name(), y='Consumption', category_orders={'x': day_names}).data[0].update(marker=dict(color='blue')),row=1, col=2)

fig_boxplot.add_trace(px.box(df_autumn, x=df_autumn.index.day_name(), y='Consumption', category_orders={'x': day_names}).data[0].update(marker=dict(color='blue')),row=2, col=1)

fig_boxplot.add_trace(px.box(df_winter, x=df_winter.index.day_name(), y='Consumption', category_orders={'x': day_names}).data[0].update(marker=dict(color='blue')),row=2, col=2)

# Update the layout
fig_boxplot.update_layout(plot_bgcolor='white')

# Display the plot
fig_boxplot.show()
#-------------------------------------------------------Print the value KPI
# Print the highest value in the "Consumption" column
highest_value_consumption = df_grouped2['Consumption'].max()
#print("Highest value:", highest_value_consumption )
print(highest_value_consumption )

# Print the lowest value in the "Consumption" column
lowest_value_consumption = df_grouped2['Consumption'].min()
#print("Lowest value:", lowest_value_consumption)
print(lowest_value_consumption)

# Print the highest value in the "Consumption" column
highest_value_production = df_grouped2['Production'].max()
#print("Highest value:", highest_value_production )
print(highest_value_production )

# Print the lowest value in the "Consumption" column
lowest_value_production = df_grouped2['Production'].min()
#print("Lowest value:", lowest_value_production)
print(lowest_value_production)

#-----------------------------------------------------------SANKEY 

df_sankey= df_grouped2.copy()

import plotly.graph_objects as go

# Filter the dataframe to only include rows with the specified date
date = '2023-03-12'
df_filtered = df_sankey[df_sankey['Date'] == date]

# Define the nodes and links for the sankey diagram
nodes = [
    {'label': 'Thermal'},
    {'label': 'Nuclear'},
    {'label': 'Wind'},
    {'label': 'Solar'},
    {'label': 'Hydro'},
    {'label': 'Pumping'},
    {'label': 'Bioenergy'},
    {'label': 'Production'},
    {'label': 'Individuals/Professionals'},
    {'label': 'Industry'},
    {'label': 'Distribution Losses'},
    #---------------------------------------------------------------
    {'label': 'Individual/Households'},
    {'label': 'Professionals/Offices/Universities'},

    {'label': 'Energy, Industry and Agriculture'},
    {'label': 'Tersier, Telecommunication, and Transport'},
    
    {'label': 'Agriculture'},
    {'label': 'Mineral'},
    {'label': 'Chemical'},
    {'label': 'Automotive'},
    {'label': 'Metallurgy and Mechanical'},
    {'label': 'Material'},
    {'label': 'Paper'},
    {'label': 'Steel'},
    {'label': 'Other'},

    {'label': 'Transport, Telecommunication'},
    {'label': 'Tersier'},

]



# Extract the label values from the nodes list
node_labels = [node['label'] for node in nodes]

links = [
    {'source': 0, 'target': 7, 'value': df_filtered['Thermal'].values[0]},
    {'source': 1, 'target': 7, 'value': df_filtered['Nuclear'].values[0]},
    {'source': 2, 'target': 7, 'value': df_filtered['Wind'].values[0]},
    {'source': 3, 'target': 7, 'value': df_filtered['Solar'].values[0]},
    {'source': 4, 'target': 7, 'value': df_filtered['Hydro'].values[0]},
    {'source': 5, 'target': 7, 'value': df_filtered['Pumping'].values[0]},
    {'source': 6, 'target': 7, 'value': df_filtered['Bioenergy'].values[0]},
    {'source': 7, 'target': 8, 'value': df_filtered['Production'].values[0]*0.4}, #Individuals/Professionals
    {'source': 7, 'target': 9, 'value': df_filtered['Production'].values[0]*0.55}, # Industry
    {'source': 7, 'target': 10, 'value': df_filtered['Production'].values[0]*0.05}, # Distribution Losses
    #-------------------------------------------------------------------------------------
    {'source': 8, 'target': 11, 'value': df_filtered['Production'].values[0]*0.3}, # Individual/Households
    {'source': 8, 'target': 12, 'value': df_filtered['Production'].values[0]*0.1}, # Professionals/Offices/Universities

    {'source': 9, 'target': 13, 'value': df_filtered['Production'].values[0]*0.35}, # Energy, Industry and Agriculture
    {'source': 9, 'target': 14, 'value': df_filtered['Production'].values[0]*0.2}, # Tersier, Telecommunication, and Transport
 
    {'source': 13, 'target': 15, 'value': df_filtered['Production'].values[0]*0.035}, # Agriculture
    {'source': 13, 'target': 16, 'value': df_filtered['Production'].values[0]*0.035}, # Mineral
    {'source': 13, 'target': 17, 'value': df_filtered['Production'].values[0]*0.035}, # Chemical
    {'source': 13, 'target': 18, 'value': df_filtered['Production'].values[0]*0.035}, # Automotive
    {'source': 13, 'target': 19, 'value': df_filtered['Production'].values[0]*0.035}, # Metallurgy
    {'source': 13, 'target': 20, 'value': df_filtered['Production'].values[0]*0.035}, #Material
    {'source': 13, 'target': 21, 'value': df_filtered['Production'].values[0]*0.035}, # Paper
    {'source': 13, 'target': 22, 'value': df_filtered['Production'].values[0]*0.035}, #Steel
    {'source': 13, 'target': 23, 'value': df_filtered['Production'].values[0]*0.07}, # Other

    {'source': 14, 'target': 24, 'value': df_filtered['Production'].values[0]*0.05}, #Transport, Telecommunication
    {'source': 14, 'target': 25, 'value': df_filtered['Production'].values[0]*0.15}, #Tersier

]

# Create the sankey diagram
fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_labels  # Use the extracted node labels
    ),
    link=dict(
        source=[link['source'] for link in links],
        target=[link['target'] for link in links],
        value=[link['value'] for link in links]
    )
)])

fig_sankey.update_layout(
    title={
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    font=dict(size=12),
    height=600,
)

fig_sankey.show()

#------------------------------WEATHER
# Read the Data
df2 = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/auvergne%20rhone%20alpes%202013-01-01%20to%202023-03-31.csv")
df2 = df2.drop(columns=['Region','Tempmax', 'Tempmin','Feelslikemax' ,'Feelslikemin', 'Feelslike', 'Dew', 'Precipprob', 'Precipcover' ,'Preciptype', 'Snowdepth', 'Windgust', 'Winddir', 'Severerisk', 'Moonphase', 'Conditions', 'Description', 'Icon', 'Stations'])


# Fill NaN values with interpolation
df2.interpolate(inplace=True)

#----------- SUNRISE - SUNSET DURATION--------------------
# Convert Sunrise and Sunset columns to datetime data type
df2['Sunrise'] = pd.to_datetime(df2['Sunrise'])
df2['Sunset'] = pd.to_datetime(df2['Sunset'])

# Extract time from Sunrise and Sunset columns
df2['Sunrise_time'] = df2['Sunrise'].dt.strftime('%H:%M:%S')
df2['Sunset_time'] = df2['Sunset'].dt.strftime('%H:%M:%S')

# Calculate duration between Sunrise and Sunset
df2['Sunduration'] = (df2['Sunset'] - df2['Sunrise']).dt.total_seconds() / 3600

# Drop Sunrise and Sunset columns
df2 = df2.drop(['Sunrise', 'Sunset'], axis=1)
df2 = df2.drop(columns=['Sunrise_time', 'Sunset_time'])
#----------------------------------------------

df2['Date'] = pd.to_datetime(df2['Date'])


# Add a new column for day of the week
df2['Dayofweek'] = df2['Date'].dt.day_name()

# --------------------------------------Add Dayindex---------------------
def get_dayindex_map(month):
    if month in [12, 1, 2]:  # Winter season
        return {'Monday': 1, 'Tuesday': 1, 'Wednesday': 1, 'Thursday': 1, 'Friday': 1, 'Saturday': 0.8, 'Sunday': 0.8}
    elif month in [3, 4, 5]:  # Spring season
        return {'Monday': 0.8, 'Tuesday': 0.8, 'Wednesday': 0.8, 'Thursday': 0.8, 'Friday': 0.8, 'Saturday': 0.6, 'Sunday': 0.6}
    elif month in [6, 7, 8]:  # Summer season
        return {'Monday': 0.6, 'Tuesday': 0.6, 'Wednesday': 0.6, 'Thursday': 0.6, 'Friday': 0.6, 'Saturday': 0.4, 'Sunday': 0.4}
    else:  # Fall season
        return {'Monday': 0.8, 'Tuesday': 0.8, 'Wednesday': 0.8, 'Thursday': 0.8, 'Friday': 0.8, 'Saturday': 0.6, 'Sunday': 0.6}

# Apply the function to modify the Dayindex column
df2['Month'] = df2['Date'].dt.month
df2['Dayindex'] = df2.apply(lambda row: get_dayindex_map(row['Month'])[row['Dayofweek']], axis=1)
df2 = df2.drop(['Month', 'Dayofweek'], axis=1)
#----------------------------------------------------------------

# Convert multiple columns to numeric dtype
numeric_columns = ['Temperature', 'Humidity', 'Precipitation', 'Snow', 'Windspeed', 'Sealevelpressure', 'Cloudcover', 'Visibility', 'Solarradiation', 'Solarenergy', 'Sunduration', 'Dayindex']
df2[numeric_columns] = df2[numeric_columns].apply(pd.to_numeric, errors='coerce')


df2.head(5)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Convert the date column to a datetime object
df2['Date'] = pd.to_datetime(df2['Date'])

# Create the subplots
fig_weather = make_subplots(rows=4, cols=3, subplot_titles=('Temperature', 'Precipitation', 'Cloudcover', 'Sunduration', 'Humidity','Solarenergy', 'Visibility', 'Windspeed', 'Sealevelpressure', 'Solarradiation', 'Snow', 'Dayindex'))

# Add the traces to the subplots
fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Temperature'], name='Temperature',
                         line=dict(color='blue', width=2)), row=1, col=1)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Precipitation'], name='Precipitation',
                         line=dict(color='red', width=2)), row=1, col=2)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Cloudcover'], name='Cloudcover',
                         line=dict(color='green', width=2)), row=1, col=3)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Sunduration'], name='Sunduration',
                         line=dict(color='purple', width=2)), row=2, col=1)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Humidity'], name='Humidity',
                         line=dict(color='orange', width=2)), row=2, col=2)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Solarenergy'], name='Solarenergy',
                         line=dict(color='brown', width=2)), row=2, col=3)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Visibility'], name='Visibility',
                         line=dict(color='Chartreuse', width=2)), row=3, col=1)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Windspeed'], name='Windspeed',
                         line=dict(color='Coral', width=2)), row=3, col=2)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Sealevelpressure'], name='Sealevelpressure',
                         line=dict(color='Crimson', width=2)), row=3, col=3)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Solarradiation'], name='Solarradiation',
                         line=dict(color='DarkCyan', width=2)), row=4, col=1)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Snow'], name='Snow',
                         line=dict(color='DarkMagenta', width=2)), row=4, col=2)

fig_weather.add_trace(go.Scatter(x=df2['Date'], y=df2['Dayindex'], name='Dayindex',
                         line=dict(color='DarkGreen', width=2)), row=4, col=3)

#'Visibility', 'Windspeed', 'Sealevelpressure', 'Solarradiation', 'Snow', 'Snow'


# Set the axis titles
fig_weather.update_xaxes(title_text='Date', row=1, col=1)
fig_weather.update_yaxes(title_text='Temperature', title_font=dict(color='blue'), row=1, col=1)
fig_weather.update_yaxes(title_text='Precipitation', title_font=dict(color='red'), row=1, col=2)
fig_weather.update_yaxes(title_text='Cloudcover', title_font=dict(color='green'), row=1, col=3)
fig_weather.update_yaxes(title_text='Sunduration', title_font=dict(color='purple'), row=2, col=1)
fig_weather.update_yaxes(title_text='Humidity', title_font=dict(color='orange'), row=2, col=2)
fig_weather.update_yaxes(title_text='Solarenergy', title_font=dict(color='brown'), row=2, col=3)
fig_weather.update_yaxes(title_text='Visibility', title_font=dict(color='Chartreuse'), row=3, col=1)
fig_weather.update_yaxes(title_text='Windspeed', title_font=dict(color='Coral'), row=3, col=2)
fig_weather.update_yaxes(title_text='Sealevelpressure', title_font=dict(color='Crimson'), row=3, col=3)
fig_weather.update_yaxes(title_text='Solarradiation', title_font=dict(color='DarkCyan'), row=4, col=1)
fig_weather.update_yaxes(title_text='Snow', title_font=dict(color='DarkMagenta'), row=4, col=2)
fig_weather.update_yaxes(title_text='Dayindex', title_font=dict(color='DarkGreen'), row=4, col=3)

# Add hover information
fig_weather.update_traces(hovertemplate='%{y:.2f}')

# Update the layout
fig_weather.update_layout(height=600, plot_bgcolor='white', showlegend=False)

# Show the figure
fig_weather.show()


#---------------------------------------FINISH

import dash
from dash import dash_table
from dash import dcc
from dash import html

external_stylesheets = ['style.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = 'Auvergne Rhone-Alpes'

# Define the layout
app.layout = html.Div(
    children=[
        # Header
        html.Div(
            children=[
                # Left column with image icon
                html.Div(
                    html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEi1b2TrokIZaf_b_sousXB9cM84AfuHI9ZLxwlnNAvx85NBN5ZSeNWQAxxwCYSXakj7guqwBKn-O1H85BpdsBMjqTQVmV9ACpLvjZmqDQ5oygps-mlWV1OxHfmm0XXJCh96RgRk0M7xkqcoPwp37A4lB1QEzFXVeSUXxdG3frpaYZ5M_RG3z-0mMYgoUg/s1600/download-removebg-preview.png"),
                    style={'width': '15%','display': 'inline-block', "text-align" :"left"}
                ),
                # Center column with title
                html.Div(
                    [
                        html.Span("Auvergne Rhone-Alpes Electricity", style={'font-family' : 'calibri', 'font-weight': '600', 'font-size': '40px', 'display': 'block'}),
                        html.Span("Author : Bahauddin Habibullah | Supervisor : Benoit Delinchant", style={'font-family' : 'calibri', 'font-size': '16px', 'display': 'block'}),
                    ],
                    style={'width': '65%', 'display': 'inline-block', 'text-align': 'left'}
                ),
                # Right column with image icon
                html.Div(
                    html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEixIt_aFT2aA2Y_FUL6N1uAAX8CW-NdlNxP6BG-ggzuhCgMdzBzeQpYb5Wb6HqtlkSentBuKjIzIY-TtlR1TPnkFyh1jSrmwyKXgUzlw0aljCT-m1O44MFo8is_tIlg59JVf4biACzqIICfONNqicCIMvA1TQzl0QlVmzkgylnfiyNVf3As0Er8jMHK0w/s1600/download__1_-removebg-preview.png"),
                    style={'width': '20%', 'display': 'inline-block', 'text-align': 'right'}
                )
            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'padding' : '40px 2%',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}
        ),

        # first row
        html.Div(
            children=[
                html.Div(
                    [html.Span("Highest Production", style={'font-family' : 'calibri', 'font-weight': '300','font-size': '20px', 'display': 'block'}),
                     #dcc.Markdown(f"Number of missing values: {df_boxplot.index.isnull().sum()}", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '40px', 'display': 'block'}),
                     dcc.Markdown(f"{highest_value_production}", style={'font-family' : 'calibri', 'font-weight': '600', 'color':'#008bd4','font-size': '40px', 'display': 'block'}),
                     #html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjTJLFaFGNCbJ7Yn67lMwndmVMKUV4kUXhY56ml8tD7OVHnuFfAyLV01p-s75NvhIKsRI0_9IfTcRyOSJ74xwpiDTQkNiatJUGBIxrpw9Ky6QgoErYXSmXD5po11R7go80HOptRZPukiWgPzekOF2EjIdG0WyUbZ9RcZPMfq0fsXsHtBmW-0wXa0xFOKg/s1600/download%20(1).png"),
                    ],
                    style ={'width': '21%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
                html.Div(
                    [html.Span("Highest Consumption", style={'font-family' : 'calibri', 'font-weight': '300',  'font-size': '20px', 'display': 'block'}),
                     #dcc.Markdown(f"Number of missing values: {df_boxplot.index.isnull().sum()}", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '40px', 'display': 'block'}),
                     dcc.Markdown(f"{highest_value_consumption}", style={'font-family' : 'calibri', 'font-weight': '600', 'color':'#008bd4','font-size': '40px', 'display': 'block'}),
                     #html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjTJLFaFGNCbJ7Yn67lMwndmVMKUV4kUXhY56ml8tD7OVHnuFfAyLV01p-s75NvhIKsRI0_9IfTcRyOSJ74xwpiDTQkNiatJUGBIxrpw9Ky6QgoErYXSmXD5po11R7go80HOptRZPukiWgPzekOF2EjIdG0WyUbZ9RcZPMfq0fsXsHtBmW-0wXa0xFOKg/s1600/download%20(1).png"),
                    ],
                    style ={'width': '21%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '1px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
                html.Div(
                    [html.Span("Lowest Production", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     #dcc.Markdown(f"Number of missing values: {df_boxplot.index.isnull().sum()}", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '40px', 'display': 'block'}),
                     dcc.Markdown(f"{lowest_value_production}", style={'font-family' : 'calibri', 'font-weight': '600', 'color':'#008bd4','font-size': '40px', 'display': 'block'}),
                     #html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjTJLFaFGNCbJ7Yn67lMwndmVMKUV4kUXhY56ml8tD7OVHnuFfAyLV01p-s75NvhIKsRI0_9IfTcRyOSJ74xwpiDTQkNiatJUGBIxrpw9Ky6QgoErYXSmXD5po11R7go80HOptRZPukiWgPzekOF2EjIdG0WyUbZ9RcZPMfq0fsXsHtBmW-0wXa0xFOKg/s1600/download%20(1).png"),
                    ],
                    style ={'width': '21%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '1px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
                html.Div(
                    [html.Span("Lowest Consumption", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     #dcc.Markdown(f"Number of missing values: {df_boxplot.index.isnull().sum()}", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '40px', 'display': 'block'}),
                     dcc.Markdown(f"{lowest_value_consumption}", style={'font-family' : 'calibri', 'font-weight': '600', 'color':'#008bd4', 'font-size': '40px', 'display': 'block'}),
                     #html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjTJLFaFGNCbJ7Yn67lMwndmVMKUV4kUXhY56ml8tD7OVHnuFfAyLV01p-s75NvhIKsRI0_9IfTcRyOSJ74xwpiDTQkNiatJUGBIxrpw9Ky6QgoErYXSmXD5po11R7go80HOptRZPukiWgPzekOF2EjIdG0WyUbZ9RcZPMfq0fsXsHtBmW-0wXa0xFOKg/s1600/download%20(1).png"),
                    ],
                    style ={'width': '21%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '1px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                )
            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}  # Set the background color to #000 (black)
        ),



        # Graphs second row
        html.Div(
            children=[
                html.Div(
                    [html.Span("Production and Consumption", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Time series of elctricity prediction and consumption over the past 8 years", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    dcc.Graph(figure=fig)
                    ],
                    style ={'width': '71%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
                html.Div(
                    [html.Span("The Region", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                    html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgaD593j9Yiignmfq2YtP_rMTKRoNeVxdZ76JXDVj0JlN3qO5kimtLIYi8zA1GXRuMWcImIAzU1h8cnNeFqbQoRZvUOreHj2CxmaM6isGIPnUyX9a79WXulfOj8sFM80gCAJvhGi-SBi6WHMTPoytA_tAQTHNP8gVUgrZVsxTaI0nZ48tOiCGrtmxODKg/s320/939px-Auvergne-Rh%C3%B4ne-Alpes_region_map_(DPJ-2020).svg.png", style={'max-width': '100%'}),
                    html.Span("Auvergne-Rhône-Alpes is a region in southeast-central France created by the 2014 territorial reform of French regions; it resulted from the merger of Auvergne and Rhône-Alpes. ", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    ],
                    style ={'width': '21%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '1px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                )
            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}  # Set the background color to #000 (black)
        ),

        # Graphs third row
        html.Div(
            children=[
                html.Div(
                    [html.Span("Electricity Allocation", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Power generation and distribution to industrial sectors", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    dcc.Graph(figure=fig_sankey)
                    ],
                    style ={'width': '46%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
                html.Div(
                    [html.Span("Weather Data", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Exogeneous Data (Weather Data)", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    dcc.Graph(figure=fig_weather)
                    ],
                    style ={'width': '46%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '1px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                )
            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}  # Set the background color to #000 (black)
        ),


        # Last row
        html.Div(
            children=[
                html.Div(
                    [html.Span("Dataset", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Dataset", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                     dash_table.DataTable(data=df_grouped2.to_dict('records'), page_size=10, 
                     #style_data={'whiteSpace': 'normal','font-family': 'calibri','height': 'auto','font-size': '10px','text-align': 'left'},
                     #style_header={'font-family': 'calibri','font-size': '11px','font-weight': 'bold','background-color': '#f3f3f3','border-top-left-radius': '10px','border-top-right-radius': '10px','padding': '5px',}
                     
                     style_table={
                        'overflowX': 'auto',
                        'width': '100%',
                        'height': '400px',
                        'margin-top': '10px',
                        'margin-bottom': '10px',
                        'border': '0px solid lightgray',
                        'border-radius': '0px',
                    },
                    style_cell={
                        'font-family': 'calibri',
                        'font-size': '10px',
                        'text-align': 'left',
                        'padding': '5px',
                    },
                    style_header={
                        'font-family': 'calibri',
                        'font-size': '11px',
                        'font-weight': 'bold',
                        'background-color': '#f3f3f3',
                        'border-top-left-radius': '10px',
                        'border-top-right-radius': '10px',
                        'padding': '5px',
                    },),
                    
                    
                    ],
                    style ={'width': '46%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
                html.Div(
                    dcc.Graph(figure=fig_boxplot),
                    style ={'width': '46%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '1px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                )
            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}  # Set the background color to #000 (black)
        ),
        
    ],
    style={'background-color': '#fcfcfc', 'margin' :'-8px'}
)

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=False, host="0.0.0.0", port=8080)


