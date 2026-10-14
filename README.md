
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

## Model Evaluation

The current model achieved 87.5% accuracy on the test set of 16 records.

The dataset is relatively small, so further testing would be needed to determine how well the model performs on new council service requests.

## Current Progress

Data preparation, model training, evaluation, example predictions, and model saving have been implemented. The API will be developed later 