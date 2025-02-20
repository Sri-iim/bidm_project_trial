import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load your dataset
@st.cache
def load_data():
    # Replace with your dataset loading logic
    data = pd.read_csv('air_quality_data.csv')
    # Convert 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])
    return data

data = load_data()

# Title and Introduction
st.title("Air Quality Analysis in India")
st.image("air_quality_image.jpg", use_column_width=True)
st.video("https://www.youtube.com/embed/your_video_id")

# Display the dataset
if st.checkbox('Show raw data'):
    st.write(data)

# Filters for Interactive Map
st.sidebar.header("Filters")
city_filter = st.sidebar.multiselect('Select Cities', data['city'].unique())
aqi_range = st.sidebar.slider('Select AQI Range', int(data['aqi'].min()), int(data['aqi'].max()), (int(data['aqi'].min()), int(data['aqi'].max())))

# Apply filters
filtered_data = data[(data['city'].isin(city_filter)) & (data['aqi'].between(aqi_range[0], aqi_range[1]))]

# Interactive India Map with Bubble Plot
st.header("Interactive India Map with AQI Levels")
fig = px.scatter_geo(filtered_data, lat='latitude', lon='longitude', size='aqi', hover_name='city', scope='asia', title='AQI Levels in India')
st.plotly_chart(fig)

# Clear filters
if st.sidebar.button('Clear Filters'):
    city_filter = []
    aqi_range = (int(data['aqi'].min()), int(data['aqi'].max()))

# List View of Records
st.header("List View of Filtered Data")
if st.checkbox('Show List View'):
    st.write(filtered_data)

# Bar Graph of Cities vs AQI
st.header("Bar Graph of Cities vs AQI")
top_n = st.selectbox('Select Top N Cities', [10, 20, 'All'])
if top_n == 'All':
    top_data = data
else:
    top_data = data.nlargest(top_n, 'aqi')

fig = px.bar(top_data, x='city', y='aqi', title=f'Top {top_n} Cities by AQI')
st.plotly_chart(fig)

# Line Graph of AQI Change per City
st.header("Line Graph of AQI Change per City")
selected_city = st.selectbox('Select City', data['city'].unique())
city_data = data[data['city'] == selected_city]

fig = px.line(city_data, x='date', y='aqi', title=f'AQI Trend for {selected_city}')
st.plotly_chart(fig)

# Time Series Prediction
st.header("Time Series Prediction of AQI")
if st.checkbox('Show Time Series Prediction'):
    # Prepare data for Prophet
    prophet_data = city_data[['date', 'aqi']].rename(columns={'date': 'ds', 'aqi': 'y'})

    # Fit the model
    model = Prophet()
    model.fit(prophet_data)

    # Make future predictions
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
    st.pyplot(fig)

# Logistic Regression for Classification
st.header("Logistic Regression for AQI Classification")
if st.checkbox('Show AQI Classification'):
    # Define AQI categories
    bins = [0, 50, 100, 200, 300, 500]
    labels = ['Good', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    data['AQI_Category'] = pd.cut(data['aqi'], bins=bins, labels=labels)

    # Prepare data for logistic regression
    X = data[['aqi']]
    y = data['AQI_Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and visualize
    y_pred = model.predict(X_test)
    st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"), x=labels, y=labels)
    st.plotly_chart(fig)
