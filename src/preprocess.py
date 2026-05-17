import pandas as pd
import string


# Clean text
def clean_text(text):

    # Convert to string safely
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    return text.strip()


# Load dataset
def load_dataset(file_path):

    # Read CSV
    df = pd.read_csv(file_path)

    # Remove empty rows
    df.dropna(inplace=True)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Ensure required columns exist
    required_columns = ['english', 'hindi']

    for col in required_columns:

        if col not in df.columns:

            raise Exception(
                f"Missing column: {col}"
            )

    # Clean text columns
    df['english'] = df['english'].apply(
        clean_text
    )

    df['hindi'] = df['hindi'].apply(
        clean_text
    )

    # Remove fully empty strings
    df = df[
        (df['english'] != '') &
        (df['hindi'] != '')
    ]

    return df


# Main
if __name__ == "__main__":

    print("\nLoading Dataset...\n")

    data = load_dataset(
        "data/parallel_corpus.csv"
    )

    print("Dataset Loaded Successfully!\n")

    print(data.head())

    # Save cleaned dataset
    data.to_csv(
        "data/cleaned_dataset.csv",
        index=False
    )

    print(
        "\nCleaned dataset saved successfully!"
    )