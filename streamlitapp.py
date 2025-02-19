import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("air_pollution_data.csv")
    return data

data = load_data()

# Sidebar for user input
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose a page:", 
                           ["Home", "AQI Prediction", "Heatmap", "Visualizations", "Gamified Experience"])

# Home Page
if options == "Home":
    st.title("Air Quality Index (AQI) Analysis and Prediction")
    st.write("Welcome to the Air Quality Index (AQI) Analysis and Prediction app!")
    st.write("This app allows you to explore air quality data, predict AQI, and visualize various aspects of air pollution.")
    st.write("Use the sidebar to navigate through different sections.")

# AQI Prediction Page
elif options == "AQI Prediction":
    st.title("AQI Prediction")
    st.write("Predict the Air Quality Index (AQI) based on various parameters.")

    # Feature selection for prediction
    features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2.5', 'pm10', 'nh3']
    X = data[features]
    y = data['aqi']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # User input for prediction
    st.sidebar.header("Input Parameters")
    co = st.sidebar.slider("CO", float(data['co'].min()), float(data['co'].max()), float(data['co'].mean()))
    no = st.sidebar.slider("NO", float(data['no'].min()), float(data['no'].max()), float(data['no'].mean()))
    no2 = st.sidebar.slider("NO2", float(data['no2'].min()), float(data['no2'].max()), float(data['no2'].mean()))
    o3 = st.sidebar.slider("O3", float(data['o3'].min()), float(data['o3'].max()), float(data['o3'].mean()))
    so2 = st.sidebar.slider("SO2", float(data['so2'].min()), float(data['so2'].max()), float(data['so2'].mean()))
    pm2_5 = st.sidebar.slider("PM2.5", float(data['pm2.5'].min()), float(data['pm2.5'].max()), float(data['pm2.5'].mean()))
    pm10 = st.sidebar.slider("PM10", float(data['pm10'].min()), float(data['pm10'].max()), float(data['pm10'].mean()))
    nh3 = st.sidebar.slider("NH3", float(data['nh3'].min()), float(data['nh3'].max()), float(data['nh3'].mean()))

    # Predict AQI
    input_data = np.array([co, no, no2, o3, so2, pm2_5, pm10, nh3]).reshape(1, -1)
    prediction = model.predict(input_data)

    st.write(f"Predicted AQI: **{prediction[0]:.2f}**")

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model Mean Squared Error: **{mse:.2f}**")

# Heatmap Page
elif options == "Heatmap":
    st.title("Heatmap of Air Quality Parameters")
    st.write("Explore the correlation between different air quality parameters.")

    # Correlation heatmap
    corr = data.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Visualizations Page
elif options == "Visualizations":
    st.title("Visualizations")
    st.write("Visualize various aspects of air quality data.")

    # AQI Distribution
    st.subheader("AQI Distribution")
    fig = px.histogram(data, x='aqi', nbins=50, title="AQI Distribution")
    st.plotly_chart(fig)

    # AQI vs PM2.5
    st.subheader("AQI vs PM2.5")
    fig = px.scatter(data, x='pm2.5', y='aqi', title="AQI vs PM2.5")
    st.plotly_chart(fig)

    # AQI vs PM10
    st.subheader("AQI vs PM10")
    fig = px.scatter(data, x='pm10', y='aqi', title="AQI vs PM10")
    st.plotly_chart(fig)

# Gamified Experience Page
elif options == "Gamified Experience":
    st.title("Gamified Experience")
    st.write("Engage in a gamified experience to learn more about air quality.")

    # Quiz on AQI
    st.subheader("AQI Quiz")
    st.write("Test your knowledge about AQI and air quality.")

    q1 = st.radio("What does AQI stand for?", 
                  ["Air Quality Index", "Air Quantity Indicator", "Atmospheric Quality Index"])
    if q1 == "Air Quality Index":
        st.success("Correct!")
    else:
        st.error("Incorrect. The correct answer is Air Quality Index.")

    q2 = st.radio("Which pollutant is most harmful to human health?", 
                  ["CO", "PM2.5", "O3"])
    if q2 == "PM2.5":
        st.success("Correct!")
    else:
        st.error("Incorrect. The correct answer is PM2.5.")

    q3 = st.radio("What is the safe level of AQI?", 
                  ["0-50", "51-100", "101-150"])
    if q3 == "0-50":
        st.success("Correct!")
    else:
        st.error("Incorrect. The correct answer is 0-50.")

    # Leaderboard
    st.subheader("Leaderboard")
    st.write("Top 5 users with the highest scores:")
    leaderboard = pd.DataFrame({
        "User": ["User1", "User2", "User3", "User4", "User5"],
        "Score": [95, 90, 85, 80, 75]
    })
    st.table(leaderboard)

# Run the app
if __name__ == "__main__":
    st.write("App is running...")
