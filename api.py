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
    Endpoint to fetch a prediction based on input parameters.
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
            (predictions_df["date"] == date)
        ]
        
        # Check if a matching prediction exists
        if prediction_row.empty:
            return jsonify({"error": f"No prediction available for {county} on {date}"}), 404
        
        # Extract the prediction value based on model_choice
        if model_choice == "wholesale":
            prediction = prediction_row["wholesale_prediction"].values[0]
        else:  # model_choice == "retail"
            prediction = prediction_row["retail_prediction"].values[0]
        
        # Return the prediction in JSON format
        return jsonify({
            "county": county,
            "date": date,
            "model_choice": model_choice,
            "prediction": prediction
        })
    
    except Exception as e:
        # Return any unexpected errors
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
