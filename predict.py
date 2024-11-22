# import pandas as pd
# import numpy as np
# import joblib
# import tensorflow as tf
# from datetime import datetime

# # Get the current date and time
# current_date = datetime.now().strftime("%Y-%m-%d")

# # Correct the path to the trained model
# trained_model = tf.keras.models.load_model(r"models\best_retail_model_sequence.h5")

# # Load historical data
# historical_data = pd.read_csv("historical_data.csv")


# def predict_and_save(current_date, county, reference_data, model, output_file):
#     # Ensure required columns are in reference_data
#     required_columns = [
#         "County", "Date", "Wholesale", "Retail", "Year", "Year_sin", "Year_cos", 
#         "Wholesale_lag_7", "Retail_lag_7", "Wholesale_rolling_mean_7d", "Retail_rolling_mean_7d", 
#         "Wholesale_rolling_std_7d", "Retail_rolling_std_7d", "County_2"
#     ]
#     for col in required_columns:
#         if col not in reference_data.columns:
#             raise ValueError(f"Missing required column in reference_data: {col}")

#     # Convert dates to datetime for filtering
#     reference_data["Date"] = pd.to_datetime(reference_data["Date"])
#     current_date = pd.to_datetime(current_date)

#     # Filter data for the specified county and ensure it's sorted by date
#     county_data = reference_data[reference_data["County"] == county].sort_values(by="Date")

#     # Check if we have sufficient historical data for feature generation
#     min_required_date = current_date - pd.Timedelta(days=7)
#     if county_data["Date"].min() > min_required_date:
#         print(f"Insufficient historical data for {county} on {current_date.date()}. Prediction cannot proceed.")
#         return  # Exit the function gracefully

#     # Generate lagged and rolling features
#     def add_lagged_and_rolling_features(group):
#         group["Wholesale_lag_7"] = group["Wholesale"].shift(7)
#         group["Retail_lag_7"] = group["Retail"].shift(7)
#         group["Wholesale_rolling_mean_7d"] = group["Wholesale"].shift(1).rolling(window=7).mean()
#         group["Retail_rolling_mean_7d"] = group["Retail"].shift(1).rolling(window=7).mean()
#         group["Wholesale_rolling_std_7d"] = group["Wholesale"].shift(1).rolling(window=7).std()
#         group["Retail_rolling_std_7d"] = group["Retail"].shift(1).rolling(window=7).std()
#         return group

#     county_data = add_lagged_and_rolling_features(county_data)

#     # Filter data for the current_date
#     input_row = county_data[county_data["Date"] == current_date]
#     if input_row.empty:
#         # Create a new row for `current_date`
#         new_row = {
#             "Date": current_date,
#             "County": county,
#             "Wholesale": np.nan,  # Placeholder, as this value isn't used for prediction
#             "Retail": np.nan,     # Placeholder, as this value isn't used for prediction
#             "Wholesale_lag_7": county_data["Wholesale"].iloc[-7],  # Use data from 7 days ago
#             "Retail_lag_7": county_data["Retail"].iloc[-7],
#             "Wholesale_rolling_mean_7d": county_data["Wholesale"].iloc[:-1].rolling(7).mean().iloc[-1],
#             "Retail_rolling_mean_7d": county_data["Retail"].iloc[:-1].rolling(7).mean().iloc[-1],
#             "Wholesale_rolling_std_7d": county_data["Wholesale"].iloc[:-1].rolling(7).std().iloc[-1],
#             "Retail_rolling_std_7d": county_data["Retail"].iloc[:-1].rolling(7).std().iloc[-1],
#         }
#         county_data = pd.concat([county_data, pd.DataFrame([new_row])], ignore_index=True)
#         input_row = county_data[county_data["Date"] == current_date]

#     # Apply the correct encoder for 'County_2'
#     binary_encoder = joblib.load("models/County_binary_encoder.pkl")  # Assuming you saved it as binary_encoder.pkl
#     county_encoded = binary_encoder.transform(pd.DataFrame({"County": [county]}))

#     # Set 'County_2' value
#     input_row.loc[:, "County_2"] = county_encoded.iloc[0, 5]

#     # Define the feature columns for the model
#     feature_columns = ['County_2', 'Year', 'Year_sin', 'Year_cos', 'Wholesale_lag_7', 'Retail_lag_7', 'Wholesale_rolling_mean_7d', 'Wholesale_rolling_std_7d', 'Retail_rolling_mean_7d' ,'Retail_rolling_std_7d']
#     if not all(col in input_row.columns for col in feature_columns):
#         missing_cols = [col for col in feature_columns if col not in input_row.columns]
#         raise ValueError(f"Missing feature columns: {missing_cols}")

#     # Collect the last 20 days of features
#     X_last_20_days = county_data[feature_columns].iloc[-20:]

#     # Check for NaNs in the features and handle them
#     if X_last_20_days.isna().any().any():
#         print(f"NaN values detected in the last 20 days of features for {county} on {current_date.date()}")
#         X_last_20_days = X_last_20_days.fillna(method='ffill').fillna(method='bfill')  # Fill NaNs with forward/backward fill

#     # Scale features
#     scaler_path = "models/scaler_retail_features.pkl"
#     scaler = joblib.load(scaler_path)
#     X_last_20_days_scaled = scaler.transform(X_last_20_days)

#     # Reshape into 3D for the model (1 batch, 20 timesteps, number of features)
#     X_pred = X_last_20_days_scaled.reshape((1, 20, len(feature_columns)))

#     # Run prediction using the model
#     prediction = model.predict(X_pred)

#     # Check if predictions are valid
#     if np.isnan(prediction).any():
#         print("Prediction contains NaN values.")
#         return  # Exit gracefully if NaNs are present in prediction

#     # Flatten prediction if it's in the shape (1, 20, 1)
#     prediction = prediction.flatten()

#     # Create a DataFrame for the next 20 days
#     future_dates = pd.date_range(start=current_date, periods=20, freq='D')
#     predicted_wholesale_df = pd.DataFrame({
#         "Date": future_dates,
#         "County": [county] * 20,
#         "Predicted_Wholesale": prediction
#     })

#     # Save the predictions to the specified output file
#     predicted_wholesale_df.to_csv(output_file, index=False)
#     print(f"Predictions for {county} from {current_date.date()} saved to {output_file}")

# # Example usage
# predict_and_save(
#     current_date=current_date,  # Use the current date
#     county="Nairobi",           # Example county
#     reference_data=historical_data,  # Your historical data
#     model=trained_model,        # Your trained model
#     output_file="predictions.csv"
# )





def predict_and_save(current_date, county, reference_data, model, output_file, wholesale_scalers, retail_scalers, encoder):
    """
    Predicts and saves wholesale and retail prices for a given county.
    
    Parameters:
    - current_date: str, the current date in "YYYY-MM-DD" format.
    - county: str, the name of the county for prediction.
    - reference_data: DataFrame, the historical price data.
    - model: Trained model for predictions.
    - output_file: str, the path to save predictions.
    - wholesale_scalers: Tuple, (scaler_features, scaler_target) for wholesale.
    - retail_scalers: Tuple, (scaler_features, scaler_target) for retail.
    - encoder: Encoder for county binary encoding.
    """
    # Extract scalers
    scaler_wholesale_features, scaler_wholesale_target = wholesale_scalers
    scaler_retail_features, scaler_retail_target = retail_scalers

    # Ensure required columns are in reference_data
    required_columns = [
        "County", "Date", "Wholesale", "Retail", "Year", "Year_sin", "Year_cos", 
        "Wholesale_lag_7", "Retail_lag_7", "Wholesale_rolling_mean_7d", "Retail_rolling_mean_7d", 
        "Wholesale_rolling_std_7d", "Retail_rolling_std_7d", "County_2"
    ]
    for col in required_columns:
        if col not in reference_data.columns:
            raise ValueError(f"Missing required column in reference_data: {col}")

    # Convert dates to datetime for filtering
    reference_data["Date"] = pd.to_datetime(reference_data["Date"])
    current_date = pd.to_datetime(current_date)

    # Filter data for the specified county and ensure it's sorted by date
    county_data = reference_data[reference_data["County"] == county].sort_values(by="Date")

    # Check if we have sufficient historical data for feature generation
    min_required_date = current_date - pd.Timedelta(days=7)
    if county_data["Date"].min() > min_required_date:
        print(f"Insufficient historical data for {county} on {current_date.date()}. Prediction cannot proceed.")
        return

    # Generate lagged and rolling features
    def add_lagged_and_rolling_features(group):
        group["Wholesale_lag_7"] = group["Wholesale"].shift(7)
        group["Retail_lag_7"] = group["Retail"].shift(7)
        group["Wholesale_rolling_mean_7d"] = group["Wholesale"].shift(1).rolling(window=7).mean()
        group["Retail_rolling_mean_7d"] = group["Retail"].shift(1).rolling(window=7).mean()
        group["Wholesale_rolling_std_7d"] = group["Wholesale"].shift(1).rolling(window=7).std()
        group["Retail_rolling_std_7d"] = group["Retail"].shift(1).rolling(window=7).std()
        return group

    county_data = add_lagged_and_rolling_features(county_data)

    # Filter data for the current_date
    input_row = county_data[county_data["Date"] == current_date]
    if input_row.empty:
        new_row = {
            "Date": current_date,
            "County": county,
            "Wholesale": np.nan,
            "Retail": np.nan,
            "Wholesale_lag_7": county_data["Wholesale"].iloc[-7],
            "Retail_lag_7": county_data["Retail"].iloc[-7],
            "Wholesale_rolling_mean_7d": county_data["Wholesale"].iloc[:-1].rolling(7).mean().iloc[-1],
            "Retail_rolling_mean_7d": county_data["Retail"].iloc[:-1].rolling(7).mean().iloc[-1],
            "Wholesale_rolling_std_7d": county_data["Wholesale"].iloc[:-1].rolling(7).std().iloc[-1],
            "Retail_rolling_std_7d": county_data["Retail"].iloc[:-1].rolling(7).std().iloc[-1],
        }
        county_data = pd.concat([county_data, pd.DataFrame([new_row])], ignore_index=True)
        input_row = county_data[county_data["Date"] == current_date]

    # Apply encoder for 'County_2'
    county_encoded = encoder.transform(pd.DataFrame({"County": [county]}))
    input_row.loc[:, "County_2"] = county_encoded.iloc[0, 5]

    # Define feature columns
    feature_columns = [
        'County_2', 'Year', 'Year_sin', 'Year_cos', 'Wholesale_lag_7', 'Retail_lag_7', 
        'Wholesale_rolling_mean_7d', 'Wholesale_rolling_std_7d', 'Retail_rolling_mean_7d', 'Retail_rolling_std_7d'
    ]
    X_last_20_days = county_data[feature_columns].iloc[-20:]

    # Handle NaNs in features
    if X_last_20_days.isna().any().any():
        X_last_20_days = X_last_20_days.fillna(method='ffill').fillna(method='bfill')

    # Scale features
    X_last_20_days_wholesale_scaled = scaler_wholesale_features.transform(X_last_20_days)
    X_last_20_days_retail_scaled = scaler_retail_features.transform(X_last_20_days)

    # Reshape for prediction
    X_pred_wholesale = X_last_20_days_wholesale_scaled.reshape((1, 20, len(feature_columns)))
    X_pred_retail = X_last_20_days_retail_scaled.reshape((1, 20, len(feature_columns)))

    # Predict wholesale and retail prices
    wholesale_prediction = model.predict(X_pred_wholesale).flatten()
    retail_prediction = model.predict(X_pred_retail).flatten()

    # Inverse scale predictions
    wholesale_prediction = scaler_wholesale_target.inverse_transform(wholesale_prediction.reshape(-1, 1)).flatten()
    retail_prediction = scaler_retail_target.inverse_transform(retail_prediction.reshape(-1, 1)).flatten()

    # Save predictions
    future_dates = pd.date_range(start=current_date, periods=20, freq='D')
    predicted_prices_df = pd.DataFrame({
        "Date": future_dates,
        "County": [county] * 20,
        "Predicted_Wholesale": wholesale_prediction,
        "Predicted_Retail": retail_prediction
    })
    predicted_prices_df.to_csv(output_file, index=False)
    print(f"Predictions for {county} saved to {output_file}")
