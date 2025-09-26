import unittest
import requests

API_URL = "http://localhost:8000"
TOKEN = "securetoken"

class TestAPI(unittest.TestCase):
    def test_predict(self):
        with open("sample.jpg", "rb") as f:
            files = {"file": f}
            headers = {"Authorization": f"Bearer {TOKEN}"}
            response = requests.post(f"{API_URL}/predict", files=files, headers=headers)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            print("Breed:", data["breed"], "Confidence:", data["confidence"])

    def test_feedback(self):
        data = {"image_id": "sample.jpg", "correct_breed": "Breed_1"}
        headers = {"Authorization": f"Bearer {TOKEN}"}
        response = requests.post(f"{API_URL}/feedback", data=data, headers=headers)
        self.assertEqual(response.status_code, 200)
        print(response.json())

    def test_upload(self):
        with open("sample.jpg", "rb") as f:
            files = {"file": f}
            data = {"breed": "Breed_1", "species": "cattle", "traits": "horn:curved,coat:brown,ear:long"}
            headers = {"Authorization": f"Bearer {TOKEN}"}
            response = requests.post(f"{API_URL}/upload", files=files, data=data, headers=headers)
            self.assertEqual(response.status_code, 200)
            print(response.json())

if __name__ == "__main__":
    unittest.main()
