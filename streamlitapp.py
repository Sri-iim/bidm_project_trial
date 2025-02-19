import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit UI
st.set_page_config(page_title="Super Cool Regression App", layout="wide")
st.title("📊 Super Interactive Regression Prediction App")

# Load dataset from GitHub repository
data_url = "https://raw.githubusercontent.com/your-repo/your-dataset.csv"
df = pd.read_csv(data_url)

# Convert date columns to datetime
for col in df.select_dtypes(include=['object']).columns:
    try:
        df[col] = pd.to_datetime(df[col])
    except:
        pass  # Skip columns that aren't dates

# Convert datetime columns to numerical values
for col in df.select_dtypes(include=['datetime']).columns:
    df[col + "_year"] = df[col].dt.year
    df[col + "_month"] = df[col].dt.month
    df[col + "_day"] = df[col].dt.day
    df.drop(columns=[col], inplace=True)

st.write("### Preview of Dataset")
st.dataframe(df.head())

# Feature Selection
target_col = st.selectbox("🎯 Select Target Variable", df.columns)
features = st.multiselect("🔢 Select Feature Variables", df.columns, default=[col for col in df.columns if col != target_col])

if target_col and features:
    X = df[features]
    y = df[target_col]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Selection
    model_choice = st.selectbox("🤖 Choose Regression Model", ["Linear Regression", "Lasso Regression", "Ridge Regression"])
    
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Lasso Regression":
        alpha = st.slider("🔧 Lasso Alpha", 0.01, 1.0, 0.1)
        model = Lasso(alpha=alpha)
    else:
        alpha = st.slider("🔧 Ridge Alpha", 0.01, 1.0, 0.1)
        model = Ridge(alpha=alpha)
    
    # Train Model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write("### Model Performance")
    st.metric("📉 Mean Squared Error", f"{mse:.4f}")
    st.metric("📈 R-squared", f"{r2:.4f}")
    
    # Plot Results
    st.write("### Prediction Visualization")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot(y_test, y_test, color='red', linewidth=2)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted Values")
    st.pyplot(fig)
    
    # Real-time Predictions
    st.write("### Make a Prediction")
    input_data = []
    for feature in features:
        input_data.append(st.number_input(f"Enter value for {feature}", value=float(df[feature].mean())))
    
    if st.button("🔮 Predict!"):
        prediction = model.predict([input_data])
        st.success(f"Predicted Value: {prediction[0]:.4f}")
    
    # Pollution Map
    st.write("### Pollution Heatmap")
    map_center = [df["latitude"].mean(), df["longitude"].mean()]
    pollution_map = folium.Map(location=map_center, zoom_start=10)
    
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color='red' if row[target_col] > df[target_col].quantile(0.75) else 'orange' if row[target_col] > df[target_col].quantile(0.5) else 'yellow',
            fill=True,
            fill_opacity=0.6,
            popup=f"Pollution Level: {row[target_col]}"
        ).add_to(pollution_map)
    
    folium_static(pollution_map)
