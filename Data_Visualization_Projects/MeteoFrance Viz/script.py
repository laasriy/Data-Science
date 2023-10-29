# Importing necessary libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import folium

# Importing necessary dataframes:
df = pd.read_csv('data/france.csv.gz', compression='gzip', parse_dates=['date'])
leg = pd.read_csv("data/postesSynop.csv", delimiter=";")
df['année'] = df.date.apply(lambda x: x.year)

# Renaming the column to join the dataframes:
df.rename(columns={"numer_sta": "ID"}, inplace=True)

t = pd.merge(df, leg, on="ID")

# Grouping data in the dataframe to get the average temperature in French cities:
moy = t.groupby(['année', 'ID', 'Nom', 'Latitude', 'Longitude', 'Altitude'])['t'].mean().reset_index()

# Transforming the temperature from Kelvin to Celsius (°C):
moy["t"] = moy['t'] - 273.15

# Visualizing the evolution of the average temperature:
fig = px.line(moy, x="année", y="t", color='Nom')
fig.update_layout(
    xaxis_title="Years",
    yaxis_title="Average temperature in °C",
    title="Evolution of the average temperature in °C between 1996 and 2023"
)

# Showing the graph in HTML format to be interactive:
fig.write_html('comparaison.html')

# Create a map using Folium library
carte = folium.Map()

# Iterate through the dataframe moy to add markers with temperature information
for idx, row in moy.iterrows():
    # Format the temperature information
    temperature_info = f'Temperature: {row.t:.2f}°C'

    # Create a marker with a tooltip showing the temperature
    m = folium.Marker([row.Latitude, row.Longitude], tooltip=temperature_info)
    m.add_to(carte)

# Show the map
carte.save("carte.html")