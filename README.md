
# BCIT601 Service Request Classifier

A machine learning project that classifies local council service requests into different categories using Logistic Regression.

## Features

- Importing and cleaning the provided dataset using pandas.
- Splitting the data into training and testing sets.
- Converting request text into numerical features using TF-IDF.
- Training a Logistic Regression classification model.
- Displaying predicted and actual categories.
- Evaluating the model using accuracy and a classification report.
- Testing the model with new example service requests.
- Saving the trained model and TF-IDF vectoriser using joblib.
- Providing predictions through a FastAPI backend.
- Returning prediction confidence scores.
- Automated testing of API endpoints using pytest.

## Installation

### Requirements

- Python
- pip
- Required Python packages listed in `requirements.txt`, if available.

### Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Prepare Data

Run the data preparation script to clean the provided dataset:

```bash
python scripts/prepare_data.py
```

### Train Model

Run the training script to train and evaluate the classifier:

```bash
python scripts/train_model.py
```

The script displays the evaluation results and example predictions. It also saves the trained model and TF-IDF vectoriser in the `model/` directory.

### Run the API

Start the FastAPI backend from the project root:

```bash
python -m uvicorn api.app:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

The API loads the saved model and vectoriser from disk when it starts. It does not retrain the model.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks whether the API is running. |
| `/categories` | GET | Returns the available prediction categories. |
| `/predict` | POST | Accepts service request text and returns a predicted category and confidence score. |

Example prediction request:

```json
{
    "text": "There is a large pothole on the road"
}
```

Example response:

```json
{
    "predicted_category": "road",
    "confidence": 38.86
}
```

The confidence score is returned as a percentage. It represents the model's estimated probability for its selected category, not a guarantee that the prediction is correct.

## Model Evaluation

The original Logistic Regression model using default TF-IDF settings achieved 87.5% accuracy on a test set of 16 records.

Experiments were conducted using Multinomial Naive Bayes and TF-IDF bigram features. Neither approach improved the original result, so the original Logistic Regression configuration was retained.

The dataset contains only 80 records, meaning further testing with a larger and more representative dataset would be needed to assess how well the model performs on real council service requests.

## Automated Testing

Automated API tests are included using pytest.

Run the tests from the project root:

```bash
python -m pytest tests/test_api.py -v
```

The tests check the health endpoint, available categories, prediction response and validation of missing input.

## Current Progress

Data preparation, model training, evaluation, model saving, API development and automated API testing have been implemented.

The project includes a working FastAPI backend with all three required endpoints. Further work includes completing the submission evidence and documenting the results and ethical considerations of the system.