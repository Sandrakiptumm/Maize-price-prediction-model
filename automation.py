import os
from datetime import datetime
import pandas as pd
from maize_data_scrapper import scrape_data
from functions import (
    load_and_combine_data,
    clean_columns,
    replace_missing_values,
    convert_price_columns,
    impute_missing_values,
    filter_markets,
    remove_outliers,
    export_data,
    feature_engineering_pipeline,
    train_lstm_model,
    generate_predictions
)

def automate_pipeline(
    scraped_csv_path="maize_data.csv",
    file_paths=[
        "raw data/Market Prices.xls", "raw data/Market Prices 2.xls",
        "raw data/Market Prices 3.xls", "raw data/Market Prices 4.xls",
        "raw data/Market Prices 5.xls", "raw data/Market Prices 6.xls",
        "raw data/Market Prices 7.xls", "raw data/Market Prices 8.xls"
    ],
    price_columns=["Wholesale", "Retail"],
    knn_columns=["Supply Volume", "Retail", "Wholesale"],
    num_columns=["Retail", "Wholesale", "Supply Volume"],
    modeling_data_path="modeling_data_2.csv",
    forecast_horizon=20,
    counties=["County_1", "County_2", "County_3", "County_4", "County_5"],
    prediction_save_path="predictions.csv"
):
    # Step 1: Scrape data
    print("Step 1: Scraping data...")
    scrape_data()

    # Step 2: Combine and clean data
    print("Step 2: Cleaning data...")
    data = load_and_combine_data(file_paths, scraped_csv_path=scraped_csv_path)
    data = clean_columns(data)
    data = replace_missing_values(data)
    data = convert_price_columns(data, price_columns)
    data = impute_missing_values(data, knn_columns)
    data = data.dropna()
    data.sort_values(by=['County', 'Market', 'Classification', 'Date'], inplace=True)
    data = filter_markets(data, threshold=10)
    data = remove_outliers(data, num_columns)
    data = data.drop_duplicates()

    # Save cleaned data
    clean_data_path = "clean_data.csv"
    export_data(data, file_name=clean_data_path)

    # Step 3: Perform feature engineering
    print("Step 3: Feature engineering...")
    feature_engineering_pipeline()

    # Step 4: Retrain models
    print("Step 4: Retraining models...")
    # Wholesale model
    train_lstm_model(
        target_variable='Wholesale',
        scaler_feature_path="models/scaler_wholesale_features.pkl",
        scaler_target_path="models/scaler_wholesale_target.pkl",
        model_save_path="models/best_wholesale_model_sequence.h5"
    )
    # Retail model
    train_lstm_model(
        target_variable='Retail',
        scaler_feature_path="models/scaler_retail_features.pkl",
        scaler_target_path="models/scaler_retail_target.pkl",
        model_save_path="models/best_retail_model_sequence.h5"
    )

    # Step 5: Generate predictions
    print("Step 5: Generating predictions...")
    generate_predictions(
        counties=counties,
        forecast_horizon=forecast_horizon,
        save_path=prediction_save_path
    )

    print("Automation complete! Predictions saved at:", prediction_save_path)


if __name__ == "__main__":
    # Run the automation pipeline
    start_time = datetime.now()
    print("Automation started at:", start_time)
    automate_pipeline()
    end_time = datetime.now()
    print("Automation finished at:", end_time)
    print("Total duration:", end_time - start_time)
