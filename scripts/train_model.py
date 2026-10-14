import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the cleaned dataset
df = pd.read_csv("./data/processed/service_requests_clean.csv")

# Select input and target columns
X = df["request_text"]
y = df["category"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Convert request text into numerical features
vectoriser = TfidfVectorizer()
X_train_tfidf = vectoriser.fit_transform(X_train)
X_test_tfidf = vectoriser.transform(X_test)

# Train the classification model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions using the testing data
predictions = model.predict(X_test_tfidf)

# Display actual and predicted categories
results = pd.DataFrame({
    "request_text": X_test,
    "actual_category": y_test,
    "predicted_category": predictions
})

print("\nPredicted Results:")
print(results)

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("Accuracy:", round(accuracy * 100, 2), "%")