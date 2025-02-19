import streamlit as st
import pandas as pd
import io

st.title("Air Pollution Data Analysis")

# Load data from GitHub
data = pd.read_csv("air_pollution_data.csv")

# Display the first few rows
st.subheader("First Few Rows of Data")
st.write(data.head())

# Get summary statistics
st.subheader("Summary Statistics")
st.write(data.describe())

# Check data types and missing values
st.subheader("Data Information")

# ✅ Fix: Use StringIO to capture data.info()
buffer = io.StringIO()
data.info(buf=buffer)  # Correctly capture the output
info_str = buffer.getvalue()  # Retrieve the text
st.text(info_str)  # Display in Streamlit
