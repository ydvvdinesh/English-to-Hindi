import pandas as pd

# Load CSV file
df = pd.read_csv("data/parallel_corpus.csv")

print("Before Cleaning:\n")
print(df.head())

# Remove empty/vacant rows
df.dropna(inplace=True)

# Remove rows having empty strings
df = df[
    (df['english'].str.strip() != '') &
    (df['hindi'].str.strip() != '')
]

print("\nAfter Removing Empty Rows:\n")
print(df.head())

# Save cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")