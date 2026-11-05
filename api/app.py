from flask import Flask, request, jsonify
import joblib
from pathlib import Path

# Create the Flask application
app = Flask(__name__)

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Set the locations of the saved model files
MODEL_PATH = BASE_DIR / "model" / "service_request_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectoriser.pkl"

# Load the saved model and vectoriser
model = joblib.load(MODEL_PATH)
vectoriser = joblib.load(VECTORIZER_PATH)