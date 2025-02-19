import streamlit as st
# ... (other imports)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score

# ... (rest of the code)

def train_and_evaluate_model(X_train, y_train, model_choice, alpha=None):
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Lasso Regression":
        model = Lasso(alpha=alpha)
    else:
        model = Ridge(alpha=alpha)

    # Feature Scaling (Example using StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data

    model.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')  # Use negative MSE for scoring
    rmse_scores = np.sqrt(-cv_scores)  # Convert back to RMSE
    mean_rmse = rmse_scores.mean()

    return model, scaler, mean_rmse #Return the scaler

# ... (inside the if target_col and features block)

    X_train, X_test, y_train, X_test_unscaled, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, scaler, mean_rmse = train_and_evaluate_model(X_train, y_train, model_choice, alpha)

    X_test_scaled = scaler.transform(X_test_unscaled) #Scale test data

    y_pred = model.predict(X_test_scaled)

    # ... (rest of the code)

    # Real-time Predictions (Scaled Input)
    input_data = []
    for feature in features:
        input_data.append(st.number_input(f"Enter value for {feature}", value=float(df[feature].mean())))

    if st.button("🔮 Predict!"):
        input_data_scaled = scaler.transform([input_data]) #Scale the input
        prediction = model.predict(input_data_scaled)
        st.success(f"Predicted Value: {prediction[0]:.4f}")

# ... (Pollution Map - Example using color scale)
    pollution_map = folium.Map(location=map_center, zoom_start=10)

    # Create a color map
    from branca.colormap import linear
    colormap = linear.YlOrRd_09.scale(df[target_col].min(), df[target_col].max())

    for _, row in df.iterrows():
        folium.CircleMarker(
            # ... (other parameters)
            color=colormap(row[target_col]),  # Use color map
            # ...
        ).add_to(pollution_map)

    folium_static(pollution_map)
