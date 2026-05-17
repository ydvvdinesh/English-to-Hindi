import pandas as pd
from collections import defaultdict


# Tokenize sentence
def tokenize(sentence):

    return sentence.split()


# Train HMM
def train_hmm(df):

    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    initial_counts = defaultdict(int)

    transition_totals = defaultdict(int)
    emission_totals = defaultdict(int)

    hindi_sentences = df['hindi'].tolist()
    english_sentences = df['english'].tolist()

    # Transition + Initial probabilities
    for sentence in hindi_sentences:

        words = tokenize(sentence)

        if len(words) > 0:
            initial_counts[words[0]] += 1

        for i in range(len(words) - 1):

            current_word = words[i]
            next_word = words[i + 1]

            transition_counts[current_word][next_word] += 1
            transition_totals[current_word] += 1

    # Emission probabilities
    for eng, hin in zip(english_sentences, hindi_sentences):

        eng_words = tokenize(eng)
        hin_words = tokenize(hin)

        min_len = min(len(eng_words), len(hin_words))

        for i in range(min_len):

            hindi_word = hin_words[i]
            english_word = eng_words[i]

            emission_counts[hindi_word][english_word] += 1
            emission_totals[hindi_word] += 1

    # Normalize transition probabilities
    transition_probs = {}

    for current_word in transition_counts:

        transition_probs[current_word] = {}

        for next_word in transition_counts[current_word]:

            transition_probs[current_word][next_word] = (
                transition_counts[current_word][next_word]
                / transition_totals[current_word]
            )

    # Normalize emission probabilities
    emission_probs = {}

    for hindi_word in emission_counts:

        emission_probs[hindi_word] = {}

        for english_word in emission_counts[hindi_word]:

            emission_probs[hindi_word][english_word] = (
                emission_counts[hindi_word][english_word]
                / emission_totals[hindi_word]
            )

    # Normalize initial probabilities
    initial_probs = {}

    total_sentences = len(hindi_sentences)

    for word in initial_counts:

        initial_probs[word] = (
            initial_counts[word]
            / total_sentences
        )

    return transition_probs, emission_probs, initial_probs


# Optimized Viterbi Algorithm
def viterbi(input_sentence,
            transition_probs,
            emission_probs,
            initial_probs):

    observations = tokenize(input_sentence)

    states = list(emission_probs.keys())

    V = [{}]
    path = {}

    # Initialize
    for state in states:

        emission = emission_probs[state].get(
            observations[0], 0
        )

        # Skip impossible states
        if emission > 0:

            V[0][state] = (
                initial_probs.get(state, 0.0001)
                * emission
            )

            path[state] = [state]

    # Dynamic Programming
    for t in range(1, len(observations)):

        V.append({})
        new_path = {}

        for current_state in states:

            emission = emission_probs[current_state].get(
                observations[t], 0
            )

            # Skip impossible emissions
            if emission == 0:
                continue

            max_prob = -1
            best_state = None

            for previous_state in V[t - 1]:

                transition = transition_probs.get(
                    previous_state, {}
                ).get(current_state, 0)

                # Skip impossible transitions
                if transition == 0:
                    continue

                prob = (
                    V[t - 1][previous_state]
                    * transition
                    * emission
                )

                if prob > max_prob:

                    max_prob = prob
                    best_state = previous_state

            if best_state is not None:

                V[t][current_state] = max_prob

                new_path[current_state] = (
                    path[best_state] + [current_state]
                )

        path = new_path

        # Stop if no valid path exists
        if len(V[t]) == 0:
            return ["Translation not found"]

    # Best final state
    max_prob = -1
    best_state = None

    for state in V[-1]:

        if V[-1][state] > max_prob:

            max_prob = V[-1][state]
            best_state = state

    if best_state is None:
        return ["Translation not found"]

    return path[best_state]


if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("data/cleaned_dataset.csv")

    # Train HMM
    transition_probs, emission_probs, initial_probs = train_hmm(df)

    # User input
    sentence = input(
        "\nEnter English sentence: "
    ).lower()

    # Predict translation
    translation = viterbi(
        sentence,
        transition_probs,
        emission_probs,
        initial_probs
    )

    print("\nPredicted Hindi Translation:\n")

    print(" ".join(translation))