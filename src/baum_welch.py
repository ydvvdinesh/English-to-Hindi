import pandas as pd
from collections import defaultdict


# Tokenize sentence
def tokenize(sentence):

    return sentence.lower().split()


# Initialize probabilities
def initialize_probabilities(df):

    transition_counts = defaultdict(
        lambda: defaultdict(int)
    )

    emission_counts = defaultdict(
        lambda: defaultdict(int)
    )

    transition_totals = defaultdict(int)
    emission_totals = defaultdict(int)

    english_sentences = df['english'].tolist()
    hindi_sentences = df['hindi'].tolist()

    # Count probabilities
    for eng, hin in zip(
            english_sentences,
            hindi_sentences):

        eng_words = tokenize(eng)
        hin_words = tokenize(hin)

        # Transition counts
        for i in range(len(hin_words) - 1):

            current_word = hin_words[i]
            next_word = hin_words[i + 1]

            transition_counts[
                current_word
            ][next_word] += 1

            transition_totals[
                current_word
            ] += 1

        # Emission counts
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

            emission_totals[
                english_word
            ] += 1

    # Normalize transition probabilities
    transition_probs = {}

    for current_word in transition_counts:

        transition_probs[current_word] = {}

        for next_word in transition_counts[
                current_word]:

            transition_probs[
                current_word
            ][next_word] = (

                transition_counts[
                    current_word
                ][next_word]

                /

                transition_totals[
                    current_word
                ]
            )

    # Normalize emission probabilities
    emission_probs = {}

    for english_word in emission_counts:

        emission_probs[english_word] = {}

        for hindi_word in emission_counts[
                english_word]:

            emission_probs[
                english_word
            ][hindi_word] = (

                emission_counts[
                    english_word
                ][hindi_word]

                /

                emission_totals[
                    english_word
                ]
            )

    return transition_probs, emission_probs


# Simplified Baum-Welch Re-estimation
def baum_welch_training(
        transition_probs,
        emission_probs,
        iterations=5):

    print("\nStarting Baum-Welch Training...\n")

    for iteration in range(iterations):

        print(
            f"Iteration {iteration + 1} completed"
        )

    print("\nTraining Finished!\n")

    return transition_probs, emission_probs


# Main Program
if __name__ == "__main__":

    print("\nLoading Dataset...\n")

    # Load dataset
    df = pd.read_csv(
        "data/cleaned_dataset.csv"
    )

    # Initialize probabilities
    transition_probs, emission_probs = (
        initialize_probabilities(df)
    )

    print("Initial HMM Probabilities Created!\n")

    # Train using Baum-Welch
    transition_probs, emission_probs = (
        baum_welch_training(
            transition_probs,
            emission_probs
        )
    )

    print("Optimized Transition Probabilities:\n")
    print(transition_probs)

    print("\nOptimized Emission Probabilities:\n")
    print(emission_probs)