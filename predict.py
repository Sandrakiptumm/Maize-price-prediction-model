import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime

def predict_and_save_all(current_date, reference_data, wholesale_model, retail_model, wholesale_scalers, retail_scalers, encoder, counties, output_file):
    """
    Predicts and saves wholesale and retail prices for all counties in one CSV.
    
    Parameters:
    - current_date: str, the current date in "YYYY-MM-DD" format.
    - reference_data: DataFrame, the historical price data.
    - wholesale_model: Trained model for wholesale predictions.
    - retail_model: Trained model for retail predictions.
    - wholesale_scalers: Tuple, (scaler_features, scaler_target) for wholesale.
    - retail_scalers: Tuple, (scaler_features, scaler_target) for retail.
    - encoder: Encoder for county binary encoding.
    - counties: List of str, names of all counties.
    - output_file: str, the path to save predictions.
    """
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

    # Initialize a list to collect predictions for all counties
    all_predictions = []

    # Iterate through each county
    for county in counties:
        # Filter data for the specified county and ensure it's sorted by date
        county_data = reference_data[reference_data["County"] == county].sort_values(by="Date")

        # Check if we have sufficient historical data for feature generation
        min_required_date = current_date - pd.Timedelta(days=7)
        if county_data["Date"].min() > min_required_date:
            print(f"Insufficient historical data for {county} on {current_date.date()}. Prediction skipped.")
            continue

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
        X_last_20_days = X_last_20_days[['Year', 'Year_sin', 'Year_cos', 'Wholesale_lag_7', 'Retail_lag_7',
       'Wholesale_rolling_mean_7d', 'Retail_rolling_mean_7d',
       'Wholesale_rolling_std_7d', 'Retail_rolling_std_7d', 'County_2']]

        # Handle NaNs in features
        if X_last_20_days.isna().any().any():
            X_last_20_days = X_last_20_days.fillna(method='ffill').fillna(method='bfill')

        # Scale features
        X_last_20_days_wholesale_scaled = wholesale_scalers[0].transform(X_last_20_days)
        X_last_20_days_retail_scaled = retail_scalers[0].transform(X_last_20_days)

        # Reshape for prediction
        X_pred_wholesale = X_last_20_days_wholesale_scaled.reshape((1, 20, len(feature_columns)))
        X_pred_retail = X_last_20_days_retail_scaled.reshape((1, 20, len(feature_columns)))

        # Predict wholesale and retail prices
        wholesale_prediction = wholesale_model.predict(X_pred_wholesale).flatten()
        retail_prediction = retail_model.predict(X_pred_retail).flatten()

        # Inverse scale predictions
        wholesale_prediction = wholesale_scalers[1].inverse_transform(wholesale_prediction.reshape(-1, 1)).flatten()
        retail_prediction = retail_scalers[1].inverse_transform(retail_prediction.reshape(-1, 1)).flatten()

        # Save predictions for this county
        future_dates = pd.date_range(start=current_date, periods=20, freq='D')
        county_predictions = pd.DataFrame({
            "Date": future_dates,
            "County": [county] * 20,
            "Predicted_Wholesale": wholesale_prediction,
            "Predicted_Retail": retail_prediction
        })

        all_predictions.append(county_predictions)

    # Combine all predictions into a single DataFrame
    combined_predictions = pd.concat(all_predictions, ignore_index=True)

    # Save to CSV
    combined_predictions.to_csv(output_file, index=False)
    print(f"All predictions saved to {output_file}")


predict_and_save_all(
    current_date="2025-1-5",
    reference_data=pd.read_csv("historical_data.csv"),
    wholesale_model=tf.keras.models.load_model(r"models/best_wholesale_model_sequence.h5"),
    retail_model=tf.keras.models.load_model(r"models/best_retail_model_sequence.h5"),
    wholesale_scalers=(joblib.load("models/scaler_wholesale_features.pkl"), joblib.load("models/scaler_wholesale_target.pkl")),
    retail_scalers=(joblib.load("models/scaler_retail_features.pkl"), joblib.load("models/scaler_retail_target.pkl")),
    encoder=joblib.load("models/County_binary_encoder.pkl"),
    counties=[
    'Baringo', 'Bomet', 'Bungoma', 'Busia', 'Elgeyo-Marakwet', 'Embu',
    'Garissa', 'Homa-bay', 'Isiolo', 'Kajiado', 'Kakamega', 'Kericho',
    'Kiambu', 'Kilifi', 'Kirinyaga', 'Kisii', 'Kisumu', 'Kitui',
    'Kwale', 'Laikipia', 'Lamu', 'Machakos', 'Makueni', 'Mandera',
    'Meru', 'Migori', 'Mombasa', 'Muranga', 'Nairobi', 'Nakuru',
    'Nandi', 'Narok', 'Nyamira', 'Nyandarua', 'Nyeri', 'Samburu',
    'Siaya', 'Taita-Taveta', 'Tana-River', 'Tharaka-Nithi',
    'Trans-Nzoia', 'Turkana', 'Uasin-Gishu', 'Vihiga', 'Wajir',
    'West-Pokot'
    ],
    output_file="predictions.csv"
)