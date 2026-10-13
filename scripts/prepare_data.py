import pandas as pd

INPUT_FILE = "./data/raw/BCIT601_starter_service_requests_dataset.csv"
OUTPUT_FILE = "./data/processed/service_requests_clean.csv"


def load_data(file_path):
    """Load the CSV dataset into a DataFrame."""
    return pd.read_csv(file_path)


def inspect_data(dataframe):
    """Display basic information about the dataset."""
    print("First five rows:")
    print(dataframe.head())

    print("\nDataset information:")
    dataframe.info()

    print("\nMissing values per column:")
    print(dataframe.isna().sum())

    print("\nDuplicate full rows:")
    print(dataframe.duplicated().sum())


def clean_data(dataframe):
    """Clean request text and handle missing values."""
    dataframe = dataframe.copy()

    # Convert request text to lowercase and remove extra whitespace
    dataframe["request_text"] = (
        dataframe["request_text"].fillna("").str.strip().str.lower()
    )

    # Remove requests with no text
    dataframe = dataframe[dataframe["request_text"] != ""]

    # Remove rows without a category label
    dataframe = dataframe.dropna(subset=["category"])

    return dataframe


def save_data(dataframe, file_path):
    """Save the cleaned dataset to a CSV file."""
    dataframe.to_csv(file_path, index=False)
    print("\nProcessed dataset saved to:", file_path)
    print("Rows retained:", len(dataframe))

    print("\nCategory counts:")
    print(dataframe["category"].value_counts())


if __name__ == "__main__":
    df = load_data(INPUT_FILE)

    inspect_data(df)

    clean_df = clean_data(df)

    save_data(clean_df, OUTPUT_FILE)