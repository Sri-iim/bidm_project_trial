import streamlit as st
import pandas as pd

st.title("Air Pollution Data Analysis")

# Load data from GitHub
url = "https://github.com/Sri-iim/Group4_BIDM_Project/blob/main/air_pollution_data.csv"
data = pd.read_csv(url)

# Display the first few rows
st.subheader("First Few Rows of Data")
st.write(data.head())

# Get summary statistics
st.subheader("Summary Statistics")
st.write(data.describe())

# Check data types and missing values
st.subheader("Data Information")
buffer = []
data.info(buf=buffer.append)
info_str = "\n".join(buffer)
st.text(info_str)
