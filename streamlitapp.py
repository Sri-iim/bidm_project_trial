import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
from prophet import Prophet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
import base64

# --- Page Configuration ---
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# --- Function to Convert Image to Base64 ---
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Convert the Image to Base64 ---
image_base64 = get_base64_of_image("pollution.jpeg")

# --- Banner Styling with Background Image ---
st.markdown(
    f"""
    <style>
        .banner {{
            position: relative;
            width: 100%;
            height: 200px; /* Adjust height */
            background: url("data:image/jpeg;base64,{image_base64}") no-repeat center center;
            background-size: cover;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: white;
            font-size: 28px;
            font-weight: bold;
        }}
        .banner::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5); /* Dark overlay */
            border-radius: 10px;
        }}
        .banner-text {{
            position: relative;
            z-index: 1;
        }}
        .subtext {{
            text-align: center;
            font-size: 18px;
            color: #ddd;
            margin-top: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Banner with Background Image ---
st.markdown(
    """
    <div class="banner">
        <div class="banner-text">🌍 Air Quality Monitoring & Prediction Dashboard</div>
    </div>
    <p class="subtext">
        This dashboard provides insights into air quality data, including visualization, prediction, and classification.
    </p>
    """,
    unsafe_allow_html=True
)


# --- Load Data ---
@st.cache_data
def load_data():
    url = "air_pollution_data.csv"  # Replace with actual dataset
    df = pd.read_csv(url)
    df["Date"] = df["Date"].apply(lambda x: parser.parse(str(x), dayfirst=True) if pd.notnull(x) else x)
    return df

df = load_data()

# Creating a top bar container for filters
with st.container():
    col1, col2, col3 = st.columns([3, 3, 1])  # Adjust column width as needed

    with col1:
        cities = st.multiselect("Select Cities", df["City"].unique(), default=df["City"].unique())

    with col2:
        aqi_range = st.slider("Select AQI Range", 
                              min_value=int(df["AQI"].min()), 
                              max_value=int(df["AQI"].max()), 
                              value=(int(df["AQI"].min()), int(df["AQI"].max())))

    with col3:
        if st.button("Clear Filters"):
            cities = df["City"].unique()
            aqi_range = (df["AQI"].min(), df["AQI"].max())

# Apply filters
filtered_df = df[(df["City"].isin(cities)) & (df["AQI"].between(aqi_range[0], aqi_range[1]))]


# --- Define AQI Color Mapping ---
aqi_colors = {
    1: "#00FF00",  # Green (Good)
    2: "#ADFF2F",  # Yellow-Green (Moderate)
    3: "#FFD700",  # Gold (Unhealthy for Sensitive Groups)
    4: "#FF8C00",  # Orange (Unhealthy)
    5: "#8B0000"   # Dark Red (Hazardous)
}

def get_aqi_color(aqi):
    """Return AQI color based on AQI category (1 to 5)."""
    aqi_colors = {
        1: "green",   # Good
        2: "yellow",  # Moderate
        3: "orange",  # Unhealthy for sensitive groups
        4: "red",     # Unhealthy
        5: "purple"   # Hazardous
    }
    return aqi_colors.get(aqi, "gray")  # Default to gray if AQI is missing

# Function to get AQI color and dynamic opacity
def get_aqi_style(aqi):
    color = aqi_colors.get(aqi, "#808080")  # Default to gray if AQI is missing
    opacity = 0.3 + (aqi * 0.15)  # Base opacity 0.3, increases with AQI level
    return color, min(opacity, 1.0)  # Cap opacity at 1.0

# --- Create a Single Interactive Map ---
st.markdown("<p class='big-font'>🌍 Interactive India AQI Map</p>", unsafe_allow_html=True)

# Initialize the Folium map
map_center = [20.5937, 78.9629]  # Center on India
aqi_map = folium.Map(location=map_center, zoom_start=5)

# Add markers for each city with AQI data
for _, row in filtered_df.iterrows():
    color, opacity = get_aqi_style(row["AQI"])
    
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8,  # Fixed size
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=opacity,  # Adjust opacity dynamically
        popup=f"{row['City']} - AQI: {row['AQI']}"
    ).add_to(aqi_map)

# Display the combined AQI map in Streamlit
st_folium(aqi_map, key="unique_aqi_map")



# --- List View ---
st.markdown("<p class='big-font'>📋 List of AQI Records</p>", unsafe_allow_html=True)
with st.expander("View List of AQI Records"):
    st.dataframe(filtered_df)

# --- Bar Chart: Top Cities vs AQI ---
st.markdown("<p class='big-font'>📊 AQI Levels Across Cities</p>", unsafe_allow_html=True)
top_n = st.selectbox("Select Top N Cities", [10, 20, "All"], index=0)
df_sorted = filtered_df.sort_values("AQI", ascending=False)
if top_n != "All":
    df_sorted = df_sorted.head(int(top_n))

fig_bar = px.bar(df_sorted, x="City", y="AQI", color="AQI", color_continuous_scale="reds", title="Top Cities by AQI")
st.plotly_chart(fig_bar)

# --- Line Chart: AQI Trends Over Time ---
st.markdown("<p class='big-font'>📈 AQI Trend Over Time</p>", unsafe_allow_html=True)
city_selected = st.selectbox("Select City for Trend Analysis", df["City"].unique())
city_df = df[df["City"] == city_selected]
fig_area = px.area(city_df, x="Date", y="AQI", title=f"AQI Trend in {city_selected}", color_discrete_sequence=["#FF5733"])
st.plotly_chart(fig_area)


# --- Time Series Prediction ---
st.markdown("<p class='big-font'>📈 Time Series Prediction of AQI</p>", unsafe_allow_html=True)
city_pred = st.selectbox("Select City for AQI Prediction", df["City"].unique())
city_df = df[df["City"] == city_pred][["Date", "AQI"]].rename(columns={"Date": "ds", "AQI": "y"})
model = Prophet()
model.fit(city_df)
future = model.make_future_dataframe(periods=365 * 2)
forecast = model.predict(future)

st.markdown(f"<p class='big-font'>📊 Predicted AQI for {city_pred} (2024 & 2025)</p>", unsafe_allow_html=True)
fig_forecast = model.plot(forecast)
st.pyplot(fig_forecast)

fig_pred = px.line(forecast, x="ds", y="yhat", title=f"Predicted AQI in {city_pred}", labels={"ds": "Date", "yhat": "Predicted AQI"})
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
log_reg = LogisticRegression(max_iter=15000, solver='saga') 
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.metric("Model Accuracy", f"{accuracy:.2%}")


from sklearn.metrics import confusion_matrix
import seaborn as sns

st.write("### 📊 Classification Report")

# Generate classification report as a dictionary
report_dict = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)

# Convert it into a DataFrame for styling
df_report = pd.DataFrame(report_dict).T

# Format numerical values to 2 decimal places
df_report = df_report.round(2)

# Rename index labels for clarity
df_report.index = df_report.index.str.replace("_", " ").str.title()


# Use Streamlit's dataframe styling for better appearance
st.dataframe(
    df_report.style.format(precision=2).background_gradient(cmap="Blues").set_properties(**{"font-size": "14px"})
)

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.write("### 🎨 Enhanced Confusion Matrix")

# Create an expander to show/hide the confusion matrix
with st.expander("📌 Click to view Confusion Matrix"):
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Close previous plots to avoid overlap
    plt.close("all")

    # Create a **smaller** figure for compact display
    fig_cm, ax = plt.subplots(figsize=(2, 2))  # Reduced size

    # Customizing the heatmap for a polished look
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="coolwarm",  # More visually striking colormap
        linewidths=0.5,  # Reduce grid lines thickness
        linecolor="gray", 
        square=True, 
        cbar=False,  # Hide color bar to save space
        annot_kws={"size": 10, "weight": "bold"},  # Smaller annotations
        xticklabels=encoder.classes_, 
        yticklabels=encoder.classes_
    )

    # Improve axis labels and title styling
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=5)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold", labelpad=5)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=10)

    # Rotate tick labels for readability
    plt.xticks(rotation=30, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # Display the confusion matrix in Streamlit
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




# --- Filter AQI Data Section ---
st.markdown("<p class='big-font'>🔍 Filter AQI Data</p>", unsafe_allow_html=True)
city_list = ["All"] + sorted(df["City"].unique().tolist())
selected_city = st.selectbox("🏙️ Select City", city_list)
aqi_range = st.slider("🌫️ Select AQI Range", min_value=int(df["AQI"].min()), max_value=int(df["AQI"].max()), value=(1, 5))

filtered_df_map = df[(df["AQI"] >= aqi_range[0]) & (df["AQI"] <= aqi_range[1])]
if selected_city != "All":
    filtered_df_map = filtered_df_map[filtered_df_map["City"] == selected_city]

aqi_map_filtered = folium.Map(location=map_center, zoom_start=5)

for _, row in filtered_df_map.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8,
        color=get_aqi_color(row["AQI"]),
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['City']}: AQI {row['AQI']}"
    ).add_to(aqi_map_filtered)

st_folium(aqi_map_filtered,key="unique_aqi_map_filter")

# --- AQI Trends Over Time Line Chart ---
st.markdown("<p class='big-font'>📈 AQI Trends Over Time</p>", unsafe_allow_html=True)
selected_city_trend = st.selectbox("🏙️ Select City for Trend Analysis", df["City"].unique())
df_city_trend = df[df["City"] == selected_city_trend].sort_values(by="Date")

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=df_city_trend["Date"], y=df_city_trend["AQI"], marker="o", ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel("AQI Level")
ax.set_title(f"AQI Trend in {selected_city_trend} Over Time")
st.pyplot(fig)

# --- AQI Time Series Prediction Section ---
st.markdown("<p class='big-font'>📅 AQI Time Series Prediction</p>", unsafe_allow_html=True)
selected_city_forecast = st.selectbox("🏙️ Select City for Prediction", df["City"].unique())
df_forecast = df[df["City"] == selected_city_forecast][["Date", "AQI"]].dropna()
df_forecast = df_forecast.rename(columns={"Date": "ds", "AQI": "y"})
model = Prophet()
model.fit(df_forecast)
future = model.make_future_dataframe(periods=730)
forecast = model.predict(future)

fig = model.plot(forecast)
st.pyplot(fig)

# --- Visualizing AQI Predictions Section ---
st.markdown("<p class='big-font'>📊 Visualizing AQI Predictions</p>", unsafe_allow_html=True)
graph_type = st.selectbox("📈 Select Graph Type", ["Line Chart", "Bar Chart", "Bubble Map"])

if graph_type == "Line Chart":
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=forecast["ds"], y=forecast["yhat"], marker="o", ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted AQI Level")
    ax.set_title(f"Predicted AQI Levels in {selected_city_forecast} (2024-2025)")
    st.pyplot(fig)

elif graph_type == "Bar Chart":
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=forecast["ds"].dt.year, y=forecast["yhat"], ax=ax, palette="coolwarm")
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted AQI Level")
    ax.set_title(f"Yearly AQI Prediction for {selected_city_forecast}")
    st.pyplot(fig)

elif graph_type == "Bubble Map":
    st.markdown("### 🗺️ Bubble Map of Predicted AQI Levels")
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
    st_folium(map_forecast,key="unique_aqi_map_2")

