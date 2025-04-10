from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tensorflow.keras.models import load_model

# configured the flask backend 
def create_app(test_config=None):


    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        app.config.from_pyfile("config.py", silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    model_path = os.path.join(app.instance_path, 'AI_forecasting_model.keras') # Joined the model path to the local project for the keras API to load it for prediction
    print(f"Model path: {model_path}")

    model = load_model(model_path)  

    @app.route('/', methods=("GET",)) # Added the HTTP GET method to render the input.html page when the backend server starts
    def welcome():
        return render_template('input.html')
    
    @app.route('/predict', methods=["POST"])
    def predict():
        try:
            data = request.get_json() # User input data recieved as a JSON format

            if not data: # handling cases when no data is received
                return jsonify({"error": "No data received"}), 400

            features = data.get('features') # Extracting the features from the JSON data

            if not features or len(features) != 5: # Checking if the features are present and have the correct length
                return jsonify({"error": "Expected 5 input features: [Close, High, Low, Open, Volume]"}), 400
 
            single_input = np.array(features).reshape(1, 5)   # Reshaping the single input to match the model's expected input shape

            repeated_sequence = np.repeat(single_input, repeats=30, axis=0)   # Repeating the input to match the model's expected input shape

            model_input = repeated_sequence.reshape(1, 30, 5)  # Reshaping all the user input to match the model's expected input shape

            print(f"Model input shape: {model_input.shape}") # Debugging the input shape

            prediction = model.predict(model_input)  # Making the prediction using the loaded model

            return jsonify({
                "prediction": prediction.tolist() # Converting the prediction to a list for JSON serialization
            })

        except Exception as e:
            print(f"Prediction error: {str(e)}")  # Debug server-side error
            return jsonify({"error": str(e)}), 500
   
    return app

    




app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
