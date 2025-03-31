from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tensorflow.keras.models import load_model


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
    
    model_path = os.path.join(app.instance_path, 'AI_forecasting_model.keras')
    print(f"Model path: {model_path}")

    model = load_model(model_path)  

    @app.route('/', methods=("GET",))
    def welcome():
        return render_template('input.html')
    
    @app.route('/predict', methods=["POST"])
    def predict():
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "No data received"}), 400

            features = data.get('features')

            if not features or len(features) != 5:
                return jsonify({"error": "Expected 5 input features: [Close, High, Low, Open, Volume]"}), 400

            single_input = np.array(features).reshape(1, 5)  

            repeated_sequence = np.repeat(single_input, repeats=30, axis=0)  

            model_input = repeated_sequence.reshape(1, 30, 5)

            print(f"Model input shape: {model_input.shape}")

            prediction = model.predict(model_input)

            return jsonify({
                "prediction": prediction.tolist()
            })

        except Exception as e:
            print(f"Prediction error: {str(e)}")  # Debug server-side error
            return jsonify({"error": str(e)}), 500
   
    return app

    




app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
