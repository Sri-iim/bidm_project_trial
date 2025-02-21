import streamlit as st
from sklearn.linear_model import LogisticRegression  # Move this up too
from sklearn.preprocessing import OrdinalEncoder 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set page title
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# Title
st.title("🌍 Air Quality Monitoring & Prediction Dashboard")

# Display an image
st.image("pollution.jpeg")

# Embed a YouTube video
st.video("https://youtu.be/e6rglsLy1Ys?si=b2QpYdTkt5dyNuhs")

import pandas as pd
import folium
import streamlit as st
from streamlit_folium import folium_static

# Load AQI dataset (Replace with actual data source)
from dateutil import parser

def load_data():
    url = "air_pollution_data.csv"  # Replace with actual dataset
    df = pd.read_csv(url)

    # Convert 'Date' column handling mixed formats
    df["Date"] = df["Date"].apply(lambda x: parser.parse(str(x), dayfirst=True) if pd.notnull(x) else x)

    return df


df = load_data()


# Filters
st.sidebar.header("Filter Data")
cities = st.sidebar.multiselect("Select Cities", df["City"].unique(), default=df["City"].unique())
aqi_range = st.sidebar.slider("Select AQI Range", min_value=int(df["AQI"].min()), max_value=int(df["AQI"].max()), value=(int(df["AQI"].min()), int(df["AQI"].max())))

# Filter dataset
filtered_df = df[(df["City"].isin(cities)) & (df["AQI"].between(aqi_range[0], aqi_range[1]))]

# Clear filters button
if st.sidebar.button("Clear Filters"):
    cities = df["City"].unique()
    aqi_range = (df["AQI"].min(), df["AQI"].max())

# Interactive Map
st.write("### 🗺️ Air Quality Index (AQI) - India Map")
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Add AQI bubbles
for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8,
        color="red" if row["AQI"] > 300 else "orange" if row["AQI"] > 200 else "yellow",
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['City']} - AQI: {row['AQI']}"
    ).add_to(m)

# Display map
folium_static(m)

import plotly.express as px

# List view in popup or embedded table
st.write("### 📋 List of AQI Records")

if st.button("Show List View in Popup"):
    st.dataframe(filtered_df)
else:
    with st.expander("View List of AQI Records"):
        st.dataframe(filtered_df)


# Bar Chart: Top Cities vs AQI
st.write("### 📊 AQI Levels Across Cities")

top_n = st.selectbox("Select Top N Cities", [10, 20, "All"], index=0)
df_sorted = filtered_df.sort_values("AQI", ascending=False)

if top_n != "All":
    df_sorted = df_sorted.head(int(top_n))

fig_bar = px.bar(df_sorted, x="City", y="AQI", color="AQI",
                 color_continuous_scale="reds", title="Top Cities by AQI")
st.plotly_chart(fig_bar)


# Line Chart: AQI Trends Over Time
st.write("### 📈 AQI Trend Over Time")

city_selected = st.selectbox("Select City for Trend Analysis", df["City"].unique())

city_df = df[df["City"] == city_selected]
fig_line = px.line(city_df, x="Date", y="AQI", title=f"AQI Trend in {city_selected}")
st.plotly_chart(fig_line)

from prophet import Prophet

st.write("### 📈 Time Series Prediction of AQI")

# Select city for prediction
city_pred = st.selectbox("Select City for AQI Prediction", df["City"].unique())

# Filter data for selected city
city_df = df[df["City"] == city_pred][["Date", "AQI"]]
city_df = city_df.rename(columns={"Date": "ds", "AQI": "y"})

# Train Prophet model
model = Prophet()
model.fit(city_df)

# Create future dataframe
future = model.make_future_dataframe(periods=365 * 2)  # Predict for 2 years (2024 & 2025)
forecast = model.predict(future)


# Plot Forecast
st.write(f"### 📊 Predicted AQI for {city_pred} (2024 & 2025)")
fig_forecast = model.plot(forecast)
st.pyplot(fig_forecast)

# Interactive Plot with Plotly
fig_pred = px.line(forecast, x="ds", y="yhat", title=f"Predicted AQI in {city_pred}",
                   labels={"ds": "Date", "yhat": "Predicted AQI"})
st.plotly_chart(fig_pred)



# Define AQI categories
def categorize_aqi(aqi):
    if aqi == 1:
        return "Good"
    elif aqi == 2:
        return "Moderate"
    elif aqi == 3:
        return "Poor"
    elif aqi == 4:
        return "Very Poor"
    else:
        return "Severe"

# Apply categorization
df["AQI_Category"] = df["AQI"].apply(categorize_aqi)

# Encode categories into numerical labels
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["AQI_Label"] = encoder.fit_transform(df["AQI_Category"])  # ✅ Use 1D array

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.write("### 🏷️ AQI Classification using Logistic Regression")

# Select features and target
X_class = df[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]  # Example pollutants
y_class = df["AQI_Label"].values.ravel()  # ✅ Convert to 1D array

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.metric("Model Accuracy", f"{accuracy:.2%}")


# # Define AQI categories
# def categorize_aqi(aqi):
#     if aqi <= 50:
#         return "Good"
#     elif aqi <= 100:
#         return "Moderate"
#     elif aqi <= 200:
#         return "Poor"
#     elif aqi <= 300:
#         return "Very Poor"
#     else:
#         return "Severe"

# # Apply categorization
# df["AQI_Category"] = df["AQI"].apply(categorize_aqi)

# # Encode categories into numerical labels
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# df["AQI_Label"] = encoder.fit_transform(df[["AQI_Category"]])

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

# st.write("### 🏷️ AQI Classification using Logistic Regression")

# # Select features and target
# X_class = df[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]  # Example pollutants
# y_class = df["AQI_Label"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# # Train model
# log_reg = LogisticRegression()
# log_reg.fit(X_train, y_train)

# # Predictions
# y_pred = log_reg.predict(X_test)

# # Display accuracy
# accuracy = accuracy_score(y_test, y_pred)
# st.metric("Model Accuracy", f"{accuracy:.2%}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

st.write("### 📊 Classification Report")
st.text(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Confusion Matrix
import matplotlib.pyplot as plt
st.write("### 📌 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
st.pyplot(fig_cm)



st.write("### 🔍 Predict AQI Category")

# User Inputs for pollutants
input_features = []
for pollutant in X_class.columns:
    value = st.number_input(f"Enter {pollutant} value", value=float(df[pollutant].mean()))
    input_features.append(value)

if st.button("Predict AQI Category"):
    pred_category = log_reg.predict([input_features])
    category_name = encoder.inverse_transform(pred_category)[0]
    st.success(f"The predicted AQI category is: **{category_name}**")


import folium
from streamlit_folium import folium_static

st.write("## 🌍 Interactive India AQI Map")

# Define a function to determine bubble color based on AQI level
def get_aqi_color(aqi):
    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "yellow"
    elif aqi <= 200:
        return "orange"
    elif aqi <= 300:
        return "red"
    else:
        return "purple"

# Create a map centered around India
map_center = [20.5937, 78.9629]  # Approximate center of India
aqi_map = folium.Map(location=map_center, zoom_start=5)

# Add AQI bubbles to the map
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8,
        color=get_aqi_color(row["AQI"]),
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['City']}: AQI {row['AQI']}"
    ).add_to(aqi_map)

# Render the map in Streamlit
folium_static(aqi_map)


st.write("### 🔍 Filter AQI Data")

# City selection filter
city_list = ["All"] + sorted(df["City"].unique().tolist())
selected_city = st.selectbox("🏙️ Select City", city_list)

# AQI range filter
aqi_range = st.slider("🌫️ Select AQI Range", min_value=int(df["AQI"].min()), max_value=int(df["AQI"].max()), value=(50, 200))

# Filter data based on user selection
filtered_df = df[(df["AQI"] >= aqi_range[0]) & (df["AQI"] <= aqi_range[1])]
if selected_city != "All":
    filtered_df = filtered_df[filtered_df["City"] == selected_city]

# Update Map
aqi_map_filtered = folium.Map(location=map_center, zoom_start=5)

for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8,
        color=get_aqi_color(row["AQI"]),
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['City']}: AQI {row['AQI']}"
    ).add_to(aqi_map_filtered)

# Render the updated map
folium_static(aqi_map_filtered)


import matplotlib.pyplot as plt
import seaborn as sns

st.write("## 📊 AQI Levels Across Cities")

# Filter for Top N Cities
top_n = st.selectbox("🔢 Select Number of Top Cities", [10, 20, 30, "All"], index=0)

# Sort by AQI values
df_sorted = df.sort_values(by="AQI", ascending=False)
if top_n != "All":
    df_sorted = df_sorted.head(int(top_n))

# Plot Bar Graph
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=df_sorted["City"], y=df_sorted["AQI"], palette="coolwarm", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel("City")
ax.set_ylabel("AQI Level")
ax.set_title(f"Top {top_n} Cities with Highest AQI")

st.pyplot(fig)


st.write("## 📈 AQI Trends Over Time")

# City Selection
selected_city_trend = st.selectbox("🏙️ Select City for Trend Analysis", df["City"].unique())

# Filter Data for Selected City
df_city_trend = df[df["City"] == selected_city_trend].sort_values(by="Date")

# Plot Line Graph
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=df_city_trend["Date"], y=df_city_trend["AQI"], marker="o", ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel("AQI Level")
ax.set_title(f"AQI Trend in {selected_city_trend} Over Time")

st.pyplot(fig)


from prophet import Prophet

st.write("## 📅 AQI Time Series Prediction")

# City Selection for Prediction
selected_city_forecast = st.selectbox("🏙️ Select City for Prediction", df["City"].unique())

# Filter data for selected city
df_forecast = df[df["City"] == selected_city_forecast][["Date", "AQI"]].dropna()

# Prophet requires columns named "ds" (date) and "y" (target variable)
df_forecast = df_forecast.rename(columns={"Date": "ds", "AQI": "y"})

# Initialize and train the model
model = Prophet()
model.fit(df_forecast)

# Future data (2024 & 2025)
future = model.make_future_dataframe(periods=730)  # Predict next 2 years (365 x 2 days)
forecast = model.predict(future)

# Plot Forecast
fig = model.plot(forecast)
st.pyplot(fig)


st.write("## 📊 Visualizing AQI Predictions")

# Select Graph Type
graph_type = st.selectbox("📈 Select Graph Type", ["Line Chart", "Bar Chart", "Bubble Map"])

# Line Chart
if graph_type == "Line Chart":
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=forecast["ds"], y=forecast["yhat"], marker="o", ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted AQI Level")
    ax.set_title(f"Predicted AQI Levels in {selected_city_forecast} (2024-2025)")
    st.pyplot(fig)

# Bar Chart
elif graph_type == "Bar Chart":
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=forecast["ds"].dt.year, y=forecast["yhat"], ax=ax, palette="coolwarm")
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted AQI Level")
    ax.set_title(f"Yearly AQI Prediction for {selected_city_forecast}")
    st.pyplot(fig)

# Bubble Map
elif graph_type == "Bubble Map":
    from streamlit_folium import folium_static
    import folium

    st.write("### 🗺️ Bubble Map of Predicted AQI Levels")

    # Sample locations (since we don't have actual lat/lon for future)
    city_lat, city_lon = df[df["City"] == selected_city_forecast][["Latitude", "Longitude"]].values[0]

    map_forecast = folium.Map(location=[city_lat, city_lon], zoom_start=10)

    folium.CircleMarker(
        location=[city_lat, city_lon],
        radius=15,
        color="red",
        fill=True,
        fill_opacity=0.6,
        popup=f"Predicted AQI: {forecast['yhat'].iloc[-1]:.2f} ({selected_city_forecast})"
    ).add_to(map_forecast)

   

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Poor"
    elif aqi <= 300:
        return "Very Poor"
    else:
        return "Severe"

# Apply categorization
df["AQI_Category"] = df["AQI"].apply(categorize_aqi)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.write("## 🏭 AQI Classification Using Logistic Regression")

# Prepare Data
X_classification = df[["AQI"]]  # Feature
y_classification = df["AQI_Category"]  # Target

encoder = LabelEncoder()
#y_classification_encoded = encoder.fit_transform(y_classification.values.reshape(-1, 1))
y_classification_encoded = encoder.fit_transform(y_classification)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification_encoded, test_size=0.2, random_state=42)

# from sklearn.model_selection import train_test_split

# # Ensure stratified sampling to maintain class balance
# X_train, X_test, y_train, y_test = train_test_split(
#     X_classification, y_classification_encoded, 
#     test_size=0.2, random_state=42, stratify=y_classification_encoded
# )

# Train Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Accuracy Metrics
accuracy = accuracy_score(y_test, y_pred)
st.metric("📊 Classification Accuracy", f"{accuracy:.2%}")

# Show Classification Report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=encoder.classes_))


st.write("## 📊 AQI Classification Distribution")

fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x=df["AQI_Category"], palette="coolwarm", order=["Good", "Moderate", "Poor", "Very Poor", "Severe"])
ax.set_xlabel("AQI Category")
ax.set_ylabel("Count")
ax.set_title("Distribution of AQI Categories")

st.pyplot(fig)


st.write("## 🔮 Predict AQI Category")

aqi_input = st.number_input("Enter AQI Value:", min_value=0, max_value=500, step=1)
if st.button("Classify AQI"):
    predicted_category = log_reg.predict([[aqi_input]])
    st.success(f"The AQI Category is: {encoder.inverse_transform(predicted_category)[0]}")
