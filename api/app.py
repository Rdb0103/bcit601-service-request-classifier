from pathlib import Path
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi import HTTPException

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

# Define the format of prediction requests
class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)

# Check whether the API is running
@app.get("/health")
def health():
    return {"status": "ok"}


# Return the categories the model can predict
@app.get("/categories")
def get_categories():
    categories = artefacts["model"].classes_.tolist()
    return {"categories": categories}

# Predict the category of a service request
@app.post("/predict")
def predict(request: PredictRequest):

    # Prevent requests containing only spaces
    if request.text.strip() == "":
        raise HTTPException(status_code=422, detail="Text cannot be blank")

    # Load the saved model and vectoriser
    model = artefacts["model"]
    vectoriser = artefacts["vectoriser"]

    # Convert the request into TF-IDF features
    features = vectoriser.transform([request.text])

    # Generate prediction probabilities
    probabilities = model.predict_proba(features)[0]

    # Find the category with the highest probability
    best_index = probabilities.argmax()

    # Return the predicted category and confidence
    return {
        "predicted_category": str(model.classes_[best_index]),
        "confidence": round(float(probabilities[best_index]) * 100, 2)
    }