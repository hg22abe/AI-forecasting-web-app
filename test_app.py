import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import json
from app import create_app  

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('app.load_model')  
        self.mock_load_model = self.patcher.start()

        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([[123.45]])
        self.mock_load_model.return_value = self.mock_model

        self.app = create_app({'TESTING': True})
        self.client = self.app.test_client()

    def tearDown(self):
        self.patcher.stop()

    def test_welcome_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_successful_prediction(self):
        payload = {
            "features": [100.5, 105.2, 98.7, 102.1, 150000]
        }

        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("prediction", data)
        self.assertEqual(data["prediction"], [[123.45]])

    def test_missing_data(self):
        response = self.client.post('/predict', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    def test_incorrect_feature_count(self):
        payload = {
            "features": [100.5, 105.2]  
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    def test_prediction_exception(self):
        # Simulate a model failure
        self.mock_model.predict.side_effect = Exception("Something went wrong")

        payload = {
            "features": [100.5, 105.2, 98.7, 102.1, 150000]
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Something went wrong")


if __name__ == '__main__':
    unittest.main()
