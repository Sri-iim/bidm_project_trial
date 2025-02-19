# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:59:50 2025

@author: user
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static

st.title("Indian Air Pollution Checker App")
st.text('Alarming Air Quality Crisis: City-wise Breakdown of India"s Polluted Urban Centers')
st.image("Pollution.jpg")
st.video('https://youtu.be/3gbJRF6d604?si=MZ_sjw85NKgvyEA')

df = pd.read_csv("air_pollution_data.csv", encoding="utf-8")
print(df.head())  # Check first few rows
print(df.columns)  # See column names

# Dropdown to select city
cities = df['city'].unique()
selected_city = st.selectbox("Select a city", cities)

# Filter data based on selected city
filtered_data = df[df['city'] == selected_city]

# Line chart for pollution levels over time
fig = px.line(filtered_data, x="date", y="aqi", title=f"aqi Levels in {selected_city}")
st.plotly_chart(fig)

# Display raw data if user wants
if st.checkbox("Show raw data"):
    st.write(filtered_data)
    

# Sidebar Filters
city_filter = st.sidebar.multiselect("Select City:", df["city"].unique(), default=df["city"].unique())
aqi_range = st.sidebar.slider("aqi Range:", int(df["aqi"].min()), int(df["aqi"].max()), (1, 5))

# Apply filters
df_filtered = df[(df["city"].isin(city_filter)) & (df["aqi"].between(aqi_range[0], aqi_range[1]))]

# Create the Folium Map
st.subheader("📍 Air Pollution Hotspot Map")
m = folium.Map(location=[22.0, 79.0], zoom_start=5)  # Centered on India

# Color scale based on aqi
def get_color(aqi):
    if aqi <= 1:
        return "green"
    elif aqi <= 2:
        return "yellow"
    elif aqi <= 3:
        return "orange"
    elif aqi <= 4:
        return "red"
    elif aqi <= 5:
        return "purple"
    else:
        return "brown"

# Add pollution markers
for index, row in df_filtered.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=7,
        color=get_color(row["aqi"]),
        fill=True,
        fill_color=get_color(row["aqi"]),
        fill_opacity=0.7,
        popup=f"{row['city']} (aqi: {row['aqi']})"
    ).add_to(m)

# Add Heatmap Layer
heat_data = [[row["latitude"], row["longitude"], row["aqi"]] for index, row in df_filtered.iterrows()]
HeatMap(heat_data).add_to(m)

# Display Map
folium_static(m)

# Display Data Table
st.subheader("Air Quality Data Table")
st.write(df_filtered[["city", "aqi", "latitude", "longitude"]].reset_index(drop=True))



