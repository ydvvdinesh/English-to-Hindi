import pandas as pd
from collections import defaultdict


# Tokenize sentence
def tokenize(sentence):

    return sentence.lower().split()


# Train HMM
def train_hmm(df):

    emission_counts = defaultdict(
        lambda: defaultdict(int)
    )

    total_counts = defaultdict(int)

    english_sentences = df['english'].tolist()
    hindi_sentences = df['hindi'].tolist()

    # Build emission probabilities
    for eng, hin in zip(
            english_sentences,
            hindi_sentences):

        eng_words = tokenize(eng)
        hin_words = tokenize(hin)

        min_len = min(
            len(eng_words),
            len(hin_words)
        )

        for i in range(min_len):

            english_word = eng_words[i]
            hindi_word = hin_words[i]

            emission_counts[
                english_word
            ][hindi_word] += 1

            total_counts[
                english_word
            ] += 1

    # Normalize probabilities
    emission_probs = {}

    for english_word in emission_counts:

        emission_probs[
            english_word
        ] = {}

        for hindi_word in emission_counts[
                english_word]:

            emission_probs[
                english_word
            ][hindi_word] = (

                emission_counts[
                    english_word
                ][hindi_word]

                /

                total_counts[
                    english_word
                ]
            )

    return emission_probs


# Translate sentence
def translate_sentence(
        sentence,
        emission_probs):

    words = tokenize(sentence)

    translated_words = []

    for word in words:

        if word in emission_probs:

            # Best Hindi word
            best_hindi_word = max(
                emission_probs[word],
                key=emission_probs[word].get
            )

            translated_words.append(
                best_hindi_word
            )

        else:

            translated_words.append(
                "[UNK]"
            )

    return " ".join(translated_words)


# Main Program
if __name__ == "__main__":

    print("\nLoading HMM Translator...\n")

    # Load dataset
    df = pd.read_csv(
        "data/cleaned_dataset.csv"
    )

    # Train model
    emission_probs = train_hmm(df)

    print("Translator Ready!\n")

    # Continuous translation loop
    while True:

        sentence = input(
            "Enter English sentence (or type exit): "
        ).lower()

        if sentence == "exit":
            break

        translation = translate_sentence(
            sentence,
            emission_probs
        )

        print("\nHindi Translation:")
        print(translation)
        print()
        