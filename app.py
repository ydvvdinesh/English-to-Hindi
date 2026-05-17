import streamlit as st
import pandas as pd
from collections import defaultdict


# Tokenize
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

            best_hindi_word = max(
                emission_probs[word],
                key=emission_probs[word].get
            )

            translated_words.append(
                best_hindi_word
            )

        else:

            translated_words.append("[UNK]")

    return " ".join(translated_words)


# ---------------- UI ---------------- #

st.set_page_config(
    page_title="HMM Translator",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 English to Hindi Translator")
st.subheader(
    "HMM + Viterbi Based Machine Translation"
)

# Load dataset
df = pd.read_csv(
    "data/cleaned_dataset.csv"
)

# Train model
emission_probs = train_hmm(df)

# User input
english_input = st.text_input(
    "Enter English Sentence:"
)

# Translate button
if st.button("Translate"):

    translation = translate_sentence(
        english_input,
        emission_probs
    )

    st.success(
        f"Hindi Translation: {translation}"
    )