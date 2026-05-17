import pandas as pd
from collections import defaultdict


# Tokenize sentence
def tokenize(sentence):

    return sentence.lower().split()


# Transition Probabilities
def calculate_transition_probs(hindi_sentences):

    transition_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)

    for sentence in hindi_sentences:

        words = tokenize(sentence)

        for i in range(len(words) - 1):

            current_word = words[i]
            next_word = words[i + 1]

            transition_counts[current_word][next_word] += 1
            total_counts[current_word] += 1

    transition_probs = {}

    for current_word in transition_counts:

        transition_probs[current_word] = {}

        for next_word in transition_counts[current_word]:

            transition_probs[current_word][next_word] = (
                transition_counts[current_word][next_word]
                / total_counts[current_word]
            )

    return transition_probs


# Improved Emission Probabilities
def calculate_emission_probs(
        english_sentences,
        hindi_sentences):

    emission_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)

    for eng, hin in zip(
            english_sentences,
            hindi_sentences):

        eng_words = tokenize(eng)
        hin_words = tokenize(hin)

        # Associate every English word
        # with Hindi words in sentence
        for english_word in eng_words:

            for hindi_word in hin_words:

                emission_counts[english_word][hindi_word] += 1
                total_counts[english_word] += 1

    emission_probs = {}

    for english_word in emission_counts:

        emission_probs[english_word] = {}

        for hindi_word in emission_counts[english_word]:

            emission_probs[english_word][hindi_word] = (
                emission_counts[english_word][hindi_word]
                / total_counts[english_word]
            )

    return emission_probs


# Initial Probabilities
def calculate_initial_probs(hindi_sentences):

    initial_counts = defaultdict(int)

    total_sentences = len(hindi_sentences)

    for sentence in hindi_sentences:

        words = tokenize(sentence)

        if len(words) > 0:

            initial_counts[words[0]] += 1

    initial_probs = {}

    for word in initial_counts:

        initial_probs[word] = (
            initial_counts[word]
            / total_sentences
        )

    return initial_probs


# Simple Translation Function
def translate_sentence(sentence,
                       emission_probs):

    words = tokenize(sentence)

    translated_words = []

    for word in words:

        if word in emission_probs:

            # Choose highest probability Hindi word
            best_hindi_word = max(
                emission_probs[word],
                key=emission_probs[word].get
            )

            translated_words.append(
                best_hindi_word
            )

        else:

            translated_words.append("[UNK]")

    return translated_words


if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv(
        "data/cleaned_dataset.csv"
    )

    english_sentences = df['english'].tolist()
    hindi_sentences = df['hindi'].tolist()

    # Calculate probabilities
    transition_probs = calculate_transition_probs(
        hindi_sentences
    )

    emission_probs = calculate_emission_probs(
        english_sentences,
        hindi_sentences
    )

    initial_probs = calculate_initial_probs(
        hindi_sentences
    )

    print("\nHMM Training Completed!")

    # User input
    sentence = input(
        "\nEnter English sentence: "
    )

    # Translate
    translation = translate_sentence(
        sentence,
        emission_probs
    )

    print("\nPredicted Hindi Translation:\n")

    print(" ".join(translation))