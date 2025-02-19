import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from branca.colormap import linear
from datetime import datetime, timedelta
from folium.plugins import HeatMap  # Import HeatMap

# Streamlit UI
st.set_page_config(page_title="Advanced Air Pollution App", layout="wide")
st.title("📊 Advanced Air Pollution Analysis")

# Load dataset (replace with your actual path)
try:
    df = pd.read_csv("air_pollution_data.csv")
except FileNotFoundError:
    st.error("Error: 'air_pollution_data.csv' not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Data Preprocessing
for col in df.select_dtypes(include=['object']).columns:
    try:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    except (TypeError, ValueError):
        pass

for col in df.select_dtypes(include=['datetime']).columns:
    df[col + "_year"] = df[col].dt.year
    df[col + "_month"] = df[col].dt.month
    df[col + "_day"] = df[col].dt.day
    df.drop(columns=[col], inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    try:
        df[col] = LabelEncoder().fit_transform(df[col])
    except Exception as e:
        st.error(f"Encoding Error for column '{col}': {e}")
        st.stop()

df.fillna(df.mean(), inplace=True)

# --- Enhanced Features ---

# 1. City-Specific AQI Plotting
st.subheader("City-Specific AQI Trends")
if 'City' in df.columns and 'Date' in df.columns and target_col in df.columns:  # Check if columns exist
    city_options = df['City'].unique()
    selected_cities = st.multiselect("Select Cities", city_options, default=city_options[:min(3, len(city_options))])

    if selected_cities:
        fig, ax = plt.subplots(figsize=(10, 6))
        for city in selected_cities:
            city_data = df[df['City'] == city].sort_values('Date')
            if not city_data.empty:
                ax.plot(city_data['Date'], city_data[target_col], label=city)
            else:
                st.warning(f"No data found for city: {city}")

        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.set_title("AQI Trends by City")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Select cities to see their AQI trends.")
else:
    st.warning("City or Date or Target column is missing, cannot plot city-specific trends.")



# 2. Future Trend Prediction (Simplified - Replace with proper time series model)
st.subheader("Future Trend Prediction (Simplified - Use a proper time series model)")
future_days = st.number_input("Enter number of days to predict", min_value=1, value=7)

if 'Date' in df.columns and target_col in df.columns: #Check if columns exist
    if st.button("Predict Future Trends"):
        try:
            last_date = df['Date'].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
            future_df = pd.DataFrame({'Date': future_dates})

            combined_df = pd.concat([df[['Date', target_col]], future_df], ignore_index=True)
            combined_df['Date_ordinal'] = combined_df['Date'].apply(datetime.toordinal)

            X = df['Date'].apply(datetime.toordinal).values.reshape(-1, 1)
            y = df[target_col]

            model = LinearRegression()  # Replace with a proper time series model
            model.fit(X, y)

            future_predictions = model.predict(combined_df['Date_ordinal'].values.reshape(-1, 1))[-future_days:]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Date'], df[target_col], label="Historical Data")
            ax.plot(future_dates, future_predictions, label="Future Predictions", linestyle="--")
            ax.set_xlabel("Date")
            ax.set_ylabel(target_col)
            ax.set_title(f"AQI Trend Prediction (Next {future_days} Days)")
            ax.legend()
            st.pyplot(fig)
        except TypeError:
            st.error("Please ensure you have a 'Date' column and it's correctly formatted as datetime.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Date or Target column is missing, cannot perform future trend prediction.")



# --- Model Training and Evaluation ---
# Feature Selection
target_col = st.selectbox("🎯 Select Target Variable", df.columns)
available_features = [col for col in df.columns if col != target_col]
features = st.multiselect("🔢 Select Feature Variables", available_features, default=available_features[:min(3, len(available_features))])

if target_col and features:
    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, X_test_unscaled, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("🤖 Choose Regression Model", ["Linear Regression", "Lasso Regression", "Ridge Regression"])

    alpha = None
    if model_choice != "Linear Regression":
        alpha = st.slider("🔧 Alpha", 0.01, 1.0, 0.1)

    def train_and_evaluate_model(X_train, y_train, model_choice, alpha=None):
        # ... (same as before)

    model, scaler, mean_rmse = train_and_evaluate_model(X_train, y_train, model_choice, alpha)
    X_test_scaled = scaler.transform(X_test_unscaled)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("### Model Performance")
    st.metric("📉 Mean Squared Error", f"{mse:.4f}")
    st.metric("📈 R-squared", f"{r2:.4f}")
    st.metric("📏 Mean Absolute Error", f"{mae:.4f}")
    st.metric("🔄 Cross-Validated RMSE", f"{mean_rmse:.4f}")

    # Plot Results
    # ... (same as before)

    # Real-time Predictions
    st.write("### Make a Prediction")
    input_data = []
    for feature in features:
        default_value = float(df[feature].mean()) if pd.api.types.is_numeric_dtype(df[feature]) else 0
        input_data.append(st.number_input(f"Enter value for {feature}", value=default_value))

    if st.button("🔮 Predict!"):
        try:
            input_data_scaled = scaler.transform([input_data])
            prediction = model.predict(input_data_scaled)
            st.success(f"Predicted Value: {prediction[0]:.4f}")
        except ValueError:
            st.error("Invalid input values. Please check the data types and ranges.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")



    # 3. Enhanced Pollution Map (Heatmap Layer)
    st.subheader("Enhanced Pollution Heatmap")

    if 'latitude' in df.columns and 'longitude' in df.columns and target_col in df.columns
    # ... (Previous code)

    # 3. Enhanced Pollution Map (Heatmap Layer)
    st.subheader("Enhanced Pollution Heatmap")

    if 'latitude' in df.columns and 'longitude' in df.columns and target_col in df.columns:
        try:
            map_center = [df["latitude"].mean(), df["longitude"].mean()]
            pollution_map = folium.Map(location=map_center, zoom_start=8)

            # Create heatmap layer
            heat_data = [[row["latitude"], row["longitude"], row[target_col]] for _, row in df.iterrows()]
            HeatMap(heat_data, radius=8, blur=5).add_to(pollution_map)

            folium_static(pollution_map)

        except Exception as e:
            st.error(f"An error occurred while creating the map: {e}")
    else:
        st.warning("Latitude, longitude, or target column is missing. Cannot display the map.")


else:
    st.warning("Please select a target variable and at least one feature variable.")
