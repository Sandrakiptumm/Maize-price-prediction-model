# import os
# from datetime import datetime
# import pandas as pd
# from maize_data_scrapper import scrape_data
# from functions import (
#     load_and_combine_data,
#     clean_columns,
#     replace_missing_values,
#     convert_price_columns,
#     impute_missing_values,
#     filter_markets,
#     remove_outliers,
#     export_data,
#     feature_engineering_pipeline,
#     train_lstm_model,
#     generate_predictions
# )

# def automate_pipeline(
#     scraped_csv_path="maize_data.csv",
#     file_paths=[
#         "raw data/Market Prices.xls", "raw data/Market Prices 2.xls",
#         "raw data/Market Prices 3.xls", "raw data/Market Prices 4.xls",
#         "raw data/Market Prices 5.xls", "raw data/Market Prices 6.xls",
#         "raw data/Market Prices 7.xls", "raw data/Market Prices 8.xls"
#     ],
#     price_columns=["Wholesale", "Retail"],
#     knn_columns=["Supply Volume", "Retail", "Wholesale"],
#     num_columns=["Retail", "Wholesale", "Supply Volume"],
#     modeling_data_path="modeling_data_2.csv",
#     forecast_horizon=20,
#     counties=["County_1", "County_2", "County_3", "County_4", "County_5"],
#     prediction_save_path="predictions.csv"
# ):
#     # Step 1: Scrape data
#     print("Step 1: Scraping data...")
#     scrape_data()

#     # Step 2: Combine and clean data
#     print("Step 2: Cleaning data...")
#     data = load_and_combine_data(file_paths, scraped_csv_path=scraped_csv_path)
#     data = clean_columns(data)
#     data = replace_missing_values(data)
#     data = convert_price_columns(data, price_columns)
#     data = impute_missing_values(data, knn_columns)
#     data = data.dropna()
#     data.sort_values(by=['County', 'Market', 'Classification', 'Date'], inplace=True)
#     data = filter_markets(data, threshold=10)
#     data = remove_outliers(data, num_columns)
#     data = data.drop_duplicates()

#     # Save cleaned data
#     clean_data_path = "clean_data.csv"
#     export_data(data, file_name=clean_data_path)

#     # Step 3: Perform feature engineering
#     print("Step 3: Feature engineering...")
#     feature_engineering_pipeline()

#     # Step 4: Retrain models
#     print("Step 4: Retraining models...")
#     # Wholesale model
#     train_lstm_model(
#         target_variable='Wholesale',
#         scaler_feature_path="models/scaler_wholesale_features.pkl",
#         scaler_target_path="models/scaler_wholesale_target.pkl",
#         model_save_path="models/best_wholesale_model_sequence.h5"
#     )
#     # Retail model
#     train_lstm_model(
#         target_variable='Retail',
#         scaler_feature_path="models/scaler_retail_features.pkl",
#         scaler_target_path="models/scaler_retail_target.pkl",
#         model_save_path="models/best_retail_model_sequence.h5"
#     )

#     # Step 5: Generate predictions
#     print("Step 5: Generating predictions...")
#     generate_predictions(
#         counties=counties,
#         forecast_horizon=forecast_horizon,
#         save_path=prediction_save_path
#     )

#     print("Automation complete! Predictions saved at:", prediction_save_path)


# if __name__ == "__main__":
#     # Run the automation pipeline
#     start_time = datetime.now()
#     print("Automation started at:", start_time)
#     automate_pipeline()
#     end_time = datetime.now()
#     print("Automation finished at:", end_time)
#     print("Total duration:", end_time - start_time)






# import os
# from datetime import datetime
# import pandas as pd
# from maize_data_scrapper import scrape_data
# from functions import (
#     load_and_combine_data,
#     clean_columns,
#     replace_missing_values,
#     convert_price_columns,
#     impute_missing_values,
#     filter_markets,
#     remove_outliers,
#     export_data,
#     feature_engineering_pipeline,
#     train_lstm_model
# )
# from predict import predict_and_save  # Import the predict_and_save function

# def automate_pipeline(
#     scraped_csv_path="maize_data.csv",
#     file_paths=[
#         "raw data/Market Prices.xls", "raw data/Market Prices 2.xls",
#         "raw data/Market Prices 3.xls", "raw data/Market Prices 4.xls",
#         "raw data/Market Prices 5.xls", "raw data/Market Prices 6.xls",
#         "raw data/Market Prices 7.xls", "raw data/Market Prices 8.xls"
#     ],
#     price_columns=["Wholesale", "Retail"],
#     knn_columns=["Supply Volume", "Retail", "Wholesale"],
#     num_columns=["Retail", "Wholesale", "Supply Volume"],
#     modeling_data_path="modeling_data_2.csv",
#     forecast_horizon=20,
#     counties=['Baringo', 'Bomet', 'Bungoma', 'Busia', 'Elgeyo-Marakwet', 'Embu',
#        'Garissa', 'Homa-bay', 'Isiolo', 'Kajiado', 'Kakamega', 'Kericho',
#        'Kiambu', 'Kilifi', 'Kirinyaga', 'Kisii', 'Kisumu', 'Kitui',
#        'Kwale', 'Laikipia', 'Lamu', 'Machakos', 'Makueni', 'Mandera',
#        'Meru', 'Migori', 'Mombasa', 'Muranga', 'Nairobi', 'Nakuru',
#        'Nandi', 'Narok', 'Nyamira', 'Nyandarua', 'Nyeri', 'Samburu',
#        'Siaya', 'Taita-Taveta', 'Tana-River', 'Tharaka-Nithi',
#        'Trans-Nzoia', 'Turkana', 'Uasin-Gishu', 'Vihiga', 'Wajir',
#        'West-Pokot'],
#     prediction_save_path="predictions.csv"
# ):
#     # Step 1: Scrape data
#     print("Step 1: Scraping data...")
#     scrape_data()

#     # Step 2: Combine and clean data
#     print("Step 2: Cleaning data...")
#     data = load_and_combine_data(file_paths, scraped_csv_path=scraped_csv_path)
#     data = clean_columns(data)
#     data = replace_missing_values(data)
#     data = convert_price_columns(data, price_columns)
#     data = impute_missing_values(data, knn_columns)
#     data = data.dropna()
#     data.sort_values(by=['County', 'Market', 'Classification', 'Date'], inplace=True)
#     data = filter_markets(data, threshold=10)
#     data = remove_outliers(data, num_columns)
#     data = data.drop_duplicates()

#     # Save cleaned data
#     clean_data_path = "clean_data.csv"
#     export_data(data, file_name=clean_data_path)

#     # Step 3: Perform feature engineering
#     print("Step 3: Feature engineering...")
#     feature_engineering_pipeline()

#     # Step 4: Retrain models
#     print("Step 4: Retraining models...")
#     # Wholesale model
#     train_lstm_model(
#         target_variable='Wholesale',
#         scaler_feature_path="models/scaler_wholesale_features.pkl",
#         scaler_target_path="models/scaler_wholesale_target.pkl",
#         model_save_path="models/best_wholesale_model_sequence.h5"
#     )
#     # Retail model
#     train_lstm_model(
#         target_variable='Retail',
#         scaler_feature_path="models/scaler_retail_features.pkl",
#         scaler_target_path="models/scaler_retail_target.pkl",
#         model_save_path="models/best_retail_model_sequence.h5"
#     )

#     # Step 5: Generate predictions using the new predict_and_save function
#     print("Step 5: Generating predictions...")
#     from tensorflow.keras.models import load_model
#     import pickle

#     # Load models and scalers
#     wholesale_model = load_model("models/best_wholesale_model_sequence.h5")
#     retail_model = load_model("models/best_retail_model_sequence.h5")

#     with open("models/scaler_wholesale_features.pkl", "rb") as f:
#         wholesale_features_scaler = pickle.load(f)
#     with open("models/scaler_wholesale_target.pkl", "rb") as f:
#         wholesale_target_scaler = pickle.load(f)
#     with open("models/scaler_retail_features.pkl", "rb") as f:
#         retail_features_scaler = pickle.load(f)
#     with open("models/scaler_retail_target.pkl", "rb") as f:
#         retail_target_scaler = pickle.load(f)

#     with open("models/encoder.pkl", "rb") as f:
#         county_encoder = pickle.load(f)

#     # Iterate over counties and predict
#     for county in counties:
#         print(f"Generating predictions for {county}...")
#         predict_and_save(
#             current_date=datetime.now().strftime("%Y-%m-%d"),
#             county=county,
#             reference_data=pd.read_csv(clean_data_path),
#             model=wholesale_model,  # Use wholesale model; adjust for retail if needed
#             output_file=prediction_save_path.replace(".csv", f"_{county}.csv"),
#             wholesale_scalers=(wholesale_features_scaler, wholesale_target_scaler),
#             retail_scalers=(retail_features_scaler, retail_target_scaler),
#             encoder=county_encoder
#         )

#     print("Automation complete! Predictions saved for all counties.")


# if __name__ == "__main__":
#     # Run the automation pipeline
#     start_time = datetime.now()
#     print("Automation started at:", start_time)
#     automate_pipeline()
#     end_time = datetime.now()
#     print("Automation finished at:", end_time)
#     print("Total duration:", end_time - start_time)

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
    train_lstm_model
)
from predict import predict_and_save

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
    forecast_horizon=20,
    counties=['Baringo', 'Bomet', 'Bungoma', 'Busia', 'Elgeyo-Marakwet', 'Embu',
       'Garissa', 'Homa-bay', 'Isiolo', 'Kajiado', 'Kakamega', 'Kericho',
       'Kiambu', 'Kilifi', 'Kirinyaga', 'Kisii', 'Kisumu', 'Kitui',
       'Kwale', 'Laikipia', 'Lamu', 'Machakos', 'Makueni', 'Mandera',
       'Meru', 'Migori', 'Mombasa', 'Muranga', 'Nairobi', 'Nakuru',
       'Nandi', 'Narok', 'Nyamira', 'Nyandarua', 'Nyeri', 'Samburu',
       'Siaya', 'Taita-Taveta', 'Tana-River', 'Tharaka-Nithi',
       'Trans-Nzoia', 'Turkana', 'Uasin-Gishu', 'Vihiga', 'Wajir',
       'West-Pokot'],
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
    clean_data_path = "clean_data2.csv"
    export_data(data, file_name=clean_data_path)

    # Step 3: Perform feature engineering
    print("Step 3: Feature engineering...")
    feature_engineering_pipeline()

    # Step 4: Generate predictions using the new predict_and_save function
    print("Step 4: Generating predictions...")
    from tensorflow.keras.models import load_model
    import joblib

    # Load scalers and encoder
    wholesale_features_scaler = joblib.load("models/scaler_wholesale_features.pkl")
    wholesale_target_scaler = joblib.load("models/scaler_wholesale_target.pkl")
    retail_features_scaler = joblib.load("models/scaler_retail_features.pkl")
    retail_target_scaler = joblib.load("models/scaler_retail_target.pkl")
    county_encoder = joblib.load("models/County_binary_encoder.pkl")


    historical_data_path =  "historical_data.csv"

    # Iterate over counties and predict
    for county in counties:
        print(f"Generating predictions for {county}...")
        predict_and_save(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            county=county,
            reference_data=pd.read_csv(historical_data_path),
            model=None,  # Predictions will use scalers and input data; adjust as necessary
            output_file=prediction_save_path.replace(".csv", f"_{county}.csv"),
            wholesale_scalers=(wholesale_features_scaler, wholesale_target_scaler),
            retail_scalers=(retail_features_scaler, retail_target_scaler),
            encoder=county_encoder
        )

    # Step 5: Retrain models
    print("Step 5: Retraining models...")
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

    print("Automation complete! Predictions saved for all counties and models retrained.")


if __name__ == "__main__":
    # Run the automation pipeline
    start_time = datetime.now()
    print("Automation started at:", start_time)
    automate_pipeline()
    end_time = datetime.now()
    print("Automation finished at:", end_time)
    print("Total duration:", end_time - start_time)
