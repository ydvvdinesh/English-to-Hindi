import pandas as pd


# Tokenize sentence
def tokenize(sentence):

    return sentence.split()


# Build vocabulary
def build_vocab(sentences):

    vocab = set()

    for sentence in sentences:

        words = tokenize(sentence)

        vocab.update(words)

    return sorted(vocab)


# Create word-to-index mapping
def create_word_index(vocab):

    return {
        word: idx
        for idx, word in enumerate(vocab)
    }


if __name__ == "__main__":

    # Load cleaned dataset
    df = pd.read_csv("data/cleaned_dataset.csv")

    # English and Hindi sentences
    english_sentences = df['english'].tolist()
    hindi_sentences = df['hindi'].tolist()

    # Build vocabularies
    english_vocab = build_vocab(english_sentences)
    hindi_vocab = build_vocab(hindi_sentences)

    # Create mappings
    english_word2idx = create_word_index(english_vocab)
    hindi_word2idx = create_word_index(hindi_vocab)

    print("\nEnglish Vocabulary:\n")
    print(english_vocab)

    print("\nHindi Vocabulary:\n")
    print(hindi_vocab)

    print("\nEnglish Word Index:\n")
    print(english_word2idx)

    print("\nHindi Word Index:\n")
    print(hindi_word2idx)

    print("\nEnglish Vocabulary Size:", len(english_vocab))
    print("Hindi Vocabulary Size:", len(hindi_vocab))