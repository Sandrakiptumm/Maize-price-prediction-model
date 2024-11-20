import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats
from flask import Flask, jsonify, request
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
import category_encoders as ce
from scipy import stats
import os
from flask import request, jsonify
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json



def load_and_combine_data(file_paths, scraped_csv_path=None):
    """
    Load multiple Excel files, and optionally a scraped CSV file, and combine them into a single DataFrame.
    """
    # Load Excel files
    dfs = [pd.read_excel(file) for file in file_paths]
    
    # Load the scraped CSV file if provided
    if scraped_csv_path:
        dfs.append(pd.read_csv(scraped_csv_path))
    
    # Combine all data into one DataFrame
    data = pd.concat(dfs, ignore_index=True)
    return data

def clean_columns(data):
    """Drop irrelevant columns."""
    return data.drop(['Commodity', 'Grade', 'Sex'], axis=1, errors='ignore')

def replace_missing_values(data):
    """Replace hyphens with NaN to handle missing values."""
    data.replace(['-', ' - ', '- ', ' -'], np.nan, inplace=True)
    return data

def convert_price_columns(data, price_columns):
    """Convert price columns to numerical values."""
    for col in price_columns:
        data[col] = data[col].str.lower().str.replace("/kg", "").str.strip()
        data[col] = data[col].str.replace("s", "").str.strip().astype(float)
    return data

def impute_missing_values(data, columns, n_neighbors=5):
    """Use KNN imputer to fill in missing values."""
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    data[columns] = knn_imputer.fit_transform(data[columns])
    return data

def filter_markets(data, threshold=10):
    """Filter out markets with less than a specified number of records."""
    market_counts = data["Market"].value_counts()
    markets_to_keep = market_counts[market_counts >= threshold].index
    return data[data['Market'].isin(markets_to_keep)]

def remove_outliers(data, columns):
    """Remove rows with outliers based on z-score thresholding."""
    for col in columns:
        # Calculate z-scores and mark rows as outliers
        z_scores = stats.zscore(data[col].dropna())
        outliers = (np.abs(z_scores) > 3)
        data = data.loc[~outliers]
    return data

def export_data(data, file_name="clean_data.csv"):
    """Export cleaned data to a CSV file."""
    data.to_csv(file_name, index=False)
    print("Data cleaning and export complete.")

def run_data_processing():
    # Define file paths and columns
    file_paths = [
        "raw data/Market Prices.xls", "raw data/Market Prices 2.xls",
        "raw data/Market Prices 3.xls", "raw data/Market Prices 4.xls",
        "raw data/Market Prices 5.xls", "Raw Data/Market Prices 6.xls",
        "raw data/Market Prices 7.xls", "Raw Data/Market Prices 8.xls"
    ]
    price_columns = ["Wholesale", "Retail"]
    knn_columns = ["Supply Volume", "Retail", "Wholesale"]
    num_columns = ["Retail", "Wholesale", "Supply Volume"]

    # Process the data
    data = load_and_combine_data(file_paths, scraped_csv_path="maize_data.csv")
    data = clean_columns(data)
    data = replace_missing_values(data)
    data = convert_price_columns(data, price_columns)
    data = impute_missing_values(data, knn_columns)
    data = data.dropna()  # Drop rows with any remaining NaN values
    data.sort_values(by=['County', 'Market', 'Classification', 'Date'], inplace=True)
    data = filter_markets(data, threshold=10)
    data = remove_outliers(data, num_columns)
    data = data.drop_duplicates()

    # Export the cleaned data to CSV
    export_data(data, "clean_data.csv")

# feature engineering function
def load_data(file_path):
    """Load the cleaned CSV data."""
    return pd.read_csv(file_path)

def extract_time_features(data):
    """Extract year, month, day, day of the week, and quarter from date."""
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Quarter'] = data['Date'].dt.quarter
    return data

def add_cyclic_features(data):
    """Add cyclic features for month, day, day of the week, quarter, and year."""
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
    data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
    data['Quarter_sin'] = np.sin(2 * np.pi * data['Quarter'] / 4)
    data['Quarter_cos'] = np.cos(2 * np.pi * data['Quarter'] / 4)
    year_range = data['Year'].max() - data['Year'].min()
    data['Year_sin'] = np.sin(2 * np.pi * (data['Year'] - data['Year'].min()) / year_range)
    data['Year_cos'] = np.cos(2 * np.pi * (data['Year'] - data['Year'].min()) / year_range)
    return data

def add_lagged_features(data, lag_days=7):
    """Add lagged features for wholesale, retail, and supply volume."""
    for lag in [lag_days]: 
        data[f'Wholesale_lag_{lag}'] = data.groupby(['County','Market', 'Classification'])['Wholesale'].shift(lag)
        data[f'Retail_lag_{lag}'] = data.groupby(['County','Market', 'Classification'])['Retail'].shift(lag)
        data[f'Supply_Volume_lag_{lag}'] = data.groupby(['County','Market', 'Classification'])['Supply Volume'].shift(lag)
    return data

def add_rolling_features(data, rolling_windows={'7d': 7}):
    """Add rolling mean and std features for wholesale, retail, and supply volume."""
    data = data.sort_values(by=['Market', 'Classification', 'Date'])
    for window_name, window_size in rolling_windows.items():
        for column in ['Wholesale', 'Retail', 'Supply Volume']:
            data[f'{column}_rolling_mean_{window_name}'] = data.groupby(['Market', 'Classification'])[column].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
            data[f'{column}_rolling_std_{window_name}'] = data.groupby(['Market', 'Classification'])[column].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
    return data.bfill().ffill()

def encode_categorical_features(data, columns_to_encode):
    """Binary encode specified categorical columns."""
    binary_encoder = ce.BinaryEncoder(cols=columns_to_encode, return_df=True)
    return binary_encoder.fit_transform(data)

def filter_columns_by_correlation(data, target_columns, threshold=0.1):
    """Filter columns based on correlation with target columns."""
    correlation_matrix = data.corr()
    correlation_with_target = correlation_matrix[target_columns]
    filtered_columns = correlation_with_target[(correlation_with_target['Retail'].abs() > threshold) | 
                                               (correlation_with_target['Wholesale'].abs() > threshold)]
    filtered_column_names = [col for col in filtered_columns.index if col not in target_columns]
    return data[filtered_column_names + target_columns]

def export_modeling_data(data, file_name="modeling_data.csv"):
    """Export the final dataset for modeling."""
    data.to_csv(file_name, index=False)

def feature_engineering_pipeline():
    # Load and process data
    data = load_data("clean_data2.csv")
    data = extract_time_features(data)
    data = add_cyclic_features(data)
    data = add_lagged_features(data)
    data = add_rolling_features(data, rolling_windows={'7d': 7})
    
    # Encode and filter features
    data = encode_categorical_features(data, columns_to_encode=['County', 'Market', 'Classification'])
    final_data = filter_columns_by_correlation(data, target_columns=['Retail', 'Wholesale'], threshold=0.1)
    
    # Export data
    export_modeling_data(final_data, "modeling_data_2.csv")
    print("Feature engineering and export complete.")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib


def train_lstm_model(
    target_variable, 
    scaler_feature_path, 
    scaler_target_path, 
    model_save_path, 
    n_timesteps=20, 
    forecast_horizon=20
):
    # Load the data
    data = pd.read_csv("modeling_data_2.csv").drop(columns=['Date'])
    
    # Define target and features
    target = data[target_variable]
    features = data.drop(columns=['Wholesale', 'Retail'])
    
    # Split data into train, validation, and test sets (80/10/10 split)
    train_size = int(len(features) * 0.8)
    val_size = int(len(features) * 0.1)
    train_X, train_y = features[:train_size], target[:train_size]
    val_X, val_y = features[train_size:train_size + val_size], target[train_size:train_size + val_size]
    test_X, test_y = features[train_size + val_size:], target[train_size + val_size:]
    
    # Scale the data
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()
    train_X_scaled = scaler_features.fit_transform(train_X)
    train_y_scaled = scaler_target.fit_transform(train_y.values.reshape(-1, 1))
    val_X_scaled = scaler_features.transform(val_X)
    val_y_scaled = scaler_target.transform(val_y.values.reshape(-1, 1))
    test_X_scaled = scaler_features.transform(test_X)
    test_y_scaled = scaler_target.transform(test_y.values.reshape(-1, 1))
    
    # Save scalers
    joblib.dump(scaler_features, scaler_feature_path)
    joblib.dump(scaler_target, scaler_target_path)
    
    # Reshape data for LSTM
    n_features = train_X.shape[1]
    def reshape_for_lstm(X, y, n_timesteps, forecast_horizon):
        X_lstm, y_lstm = [], []
        for i in range(n_timesteps, len(X) - forecast_horizon + 1):
            X_lstm.append(X[i - n_timesteps:i])
            y_lstm.append(y[i:i + forecast_horizon])
        return np.array(X_lstm), np.array(y_lstm)
    
    train_X_lstm, train_y_lstm = reshape_for_lstm(train_X_scaled, train_y_scaled, n_timesteps, forecast_horizon)
    val_X_lstm, val_y_lstm = reshape_for_lstm(val_X_scaled, val_y_scaled, n_timesteps, forecast_horizon)
    
    # Build the LSTM model
    def build_lstm_model(n_timesteps, n_features, forecast_horizon):
        model = Sequential()
        model.add(LSTM(16, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01), input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.4))
        model.add(LSTM(16, activation='tanh', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.4))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(forecast_horizon))
        model.compile(optimizer=Adam(learning_rate=0.00005), loss='mean_squared_error')
        return model
    
    model = build_lstm_model(n_timesteps, n_features, forecast_horizon)
    
    # Configure callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
    
    # Train the model
    history = model.fit(
        train_X_lstm, train_y_lstm,
        epochs=50,
        batch_size=32,
        validation_data=(val_X_lstm, val_y_lstm),
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=2
    )
    return history


# Train Wholesale model
train_lstm_model(
    target_variable='Wholesale',
    scaler_feature_path="models/scaler_wholesale_features.pkl",
    scaler_target_path="models/scaler_wholesale_target.pkl",
    model_save_path="models/best_wholesale_model_sequence.h5"
)

# Train Retail model
train_lstm_model(
    target_variable='Retail',
    scaler_feature_path="models/scaler_retail_features.pkl",
    scaler_target_path="models/scaler_retail_target.pkl",
    model_save_path="models/best_retail_model_sequence.h5"
)


from flask import Flask
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib


# Initialize Flask app
app = Flask(__name__)

# Load pre-trained Keras models
wholesale_model = load_model("models/best_wholesale_model_sequence.h5")
retail_model = load_model("models/best_retail_model_sequence.h5")

# Load scalers
scaler_features_wholesale = joblib.load("models/scaler_wholesale_features.pkl")
scaler_wholesale_target = joblib.load("models/scaler_wholesale_target.pkl")
scaler_features_retail = joblib.load("models/scaler_retail_features.pkl")
scaler_retail_target = joblib.load("models/scaler_retail_target.pkl")

# Load binary encoder
binary_encoder = joblib.load('binary_encoder.pkl')

# Generate predictions function
def generate_predictions(
    counties, 
    forecast_horizon=20, 
    lag_features_placeholder=[0.0] * 6,  # Adjust length based on model requirements
    save_path="predictions.csv"
):
    """
    Generates wholesale and retail predictions for the next forecast_horizon days
    for the given counties and saves results to a CSV file.

    Args:
        counties (list): List of county names to predict for.
        forecast_horizon (int): Number of days to forecast.
        lag_features_placeholder (list): Placeholder for lag features (to be replaced with actual logic).
        save_path (str): Path to save the predictions CSV.
    """
    predictions = []
    today = datetime.now().date()

    # Iterate over counties and forecast horizon
    for county in counties:
        for i in range(1, forecast_horizon + 1):
            future_date = today + timedelta(days=i)
            year = future_date.year
            year_sin = np.sin(2 * np.pi * (future_date.timetuple().tm_yday / 365))
            year_cos = np.cos(2 * np.pi * (future_date.timetuple().tm_yday / 365))
            
            try:
                # Binary encode the county
                county_encoded = binary_encoder.transform(pd.DataFrame({"County": [county]}))
                # Ensure the encoder output matches model expectations
                county_encoded_values = county_encoded.values.flatten()
            except ValueError as e:
                print(f"Encoding failed for county '{county}': {e}")
                continue
            
            # Prepare features
            features = np.array([*county_encoded_values, year, year_sin, year_cos] + lag_features_placeholder).reshape(1, -1)
            
            # Wholesale prediction
            scaled_features_wholesale = scaler_features_wholesale.transform(features)
            scaled_features_wholesale = scaled_features_wholesale.reshape((1, scaled_features_wholesale.shape[1], 1))
            scaled_prediction_wholesale = wholesale_model.predict(scaled_features_wholesale)
            wholesale_prediction = scaler_wholesale_target.inverse_transform(scaled_prediction_wholesale)[0][0]
            
            # Retail prediction
            scaled_features_retail = scaler_features_retail.transform(features)
            scaled_features_retail = scaled_features_retail.reshape((1, scaled_features_retail.shape[1], 1))
            scaled_prediction_retail = retail_model.predict(scaled_features_retail)
            retail_prediction = scaler_retail_target.inverse_transform(scaled_prediction_retail)[0][0]
            
            # Save predictions
            predictions.append({
                "county": county,
                "date": str(future_date),
                "wholesale_prediction": wholesale_prediction,
                "retail_prediction": retail_prediction
            })
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}!")

