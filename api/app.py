from pathlib import Path
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Set the locations of the saved model files
MODEL_PATH = BASE_DIR / "model" / "service_request_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectoriser.pkl"


# Store the loaded model and vectoriser
artefacts = {}


# Load the model and vectoriser when the API starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    artefacts["model"] = joblib.load(MODEL_PATH)
    artefacts["vectoriser"] = joblib.load(VECTORIZER_PATH)
    yield
    artefacts.clear()


# Create the FastAPI application
app = FastAPI(
    title="Council Service Request Classifier",
    lifespan=lifespan
)