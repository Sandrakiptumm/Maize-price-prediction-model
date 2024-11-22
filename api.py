from flask import Flask, request, jsonify
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load saved predictions
try:
    predictions_df = pd.read_csv("predictions.csv")  # Ensure this file exists and is correctly formatted
except FileNotFoundError:
    raise Exception("The predictions.csv file is missing. Ensure it is present in the same directory as this script.")

@app.route("/predict", methods=["GET"])
def get_prediction():
    """
    Endpoint to fetch a single prediction based on input parameters.
    Parameters (query string):
    - county: The county name
    - date: The date for prediction in 'YYYY-MM-DD' format
    - model_choice: 'wholesale' or 'retail'
    Returns:
    - JSON response containing the prediction or an error message
    """
    try:
        # Get input parameters from the request
        county = request.args.get("county")
        date = request.args.get("date")
        model_choice = request.args.get("model_choice")  # "wholesale" or "retail"
        
        # Validate required inputs
        if not county or not date or not model_choice:
            return jsonify({"error": "Missing required fields: county, date, or model_choice"}), 400
        
        # Validate model choice
        if model_choice not in ["wholesale", "retail"]:
            return jsonify({"error": "Invalid model_choice. Choose 'wholesale' or 'retail'"}), 400
        
        # Fetch the prediction row from the DataFrame
        prediction_row = predictions_df[
            (predictions_df["county"].str.lower() == county.lower()) & 
            (predictions_df["Date"] == date)
        ]
        
        # Check if a matching prediction exists
        if prediction_row.empty:
            return jsonify({"error": f"No prediction available for {county} on {date}"}), 404
        
        # Extract the prediction value based on model_choice
        prediction = prediction_row["Predicted_Wholesale"].values[0] if model_choice == "wholesale" else None
        
        # Return the prediction in JSON format
        return jsonify({
            "county": county,
            "date": date,
            "model_choice": model_choice,
            "prediction": prediction
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def get_max_min():
    """
    Endpoint to fetch the maximum and minimum predictions for a county.
    Parameters (query string):
    - county: The county name
    Returns:
    - JSON response containing the max, min, and county or an error message
    """
    try:
        # Get input parameter from the request
        county = request.args.get("county")
        
        # Validate required input
        if not county:
            return jsonify({"error": "Missing required field: county"}), 400
        
        # Filter predictions for the specified county
        county_data = predictions_df[predictions_df["County"].str.lower() == county.lower()]
        
        if county_data.empty:
            return jsonify({"error": f"No predictions available for {county}"}), 404
        
        # Compute max and min predicted wholesale prices
        max_price = county_data["Predicted_Wholesale"].max()
        min_price = county_data["Predicted_Wholesale"].min()
        
        return jsonify({
            "county": county,
            "max_predicted_wholesale": max_price,
            "min_predicted_wholesale": min_price
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
