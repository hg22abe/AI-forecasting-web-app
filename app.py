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
    
    @app.route('/predict', methods=("POST",))
    def predict():
        try:
            data = request.get_json()

            if data is None or 'features' not in data:
                return jsonify({"error": "No input features provided"}), 400

            features = np.array(data['features']).reshape(1, -1)  

            prediction = model.predict(features)

            return jsonify({
                "prediction": prediction.tolist()  
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
   
    return app

    




app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
