# English → Hindi HMM Translation System

This project implements an **English-to-Hindi machine translation system using a Hidden Markov Model (HMM)**. The system follows a structured pipeline divided into two major phases: **Training** and **Prediction (Inference)**.

## 🔄 Complete Workflow

```text
preprocess.py
      ↓
tokenizer.py
      ↓
train_hmm.py
      ↓
baum_welch.py
      ↓
viterbi.py
      ↓
translator.py
      ↓
app.py
```

The first four files prepare and train the translation model, while the final three files use the trained model to generate Hindi translations.

---

## 🧠 Training Phase

The following files are responsible for preparing the data and training the HMM:

### 1️⃣ `preprocess.py`

**Purpose:** Dataset preprocessing and cleaning.

This script:

* Cleans the translation dataset
* Removes empty or invalid rows
* Prepares the data for tokenization
* Generates `cleaned_dataset.csv`

**Run first:**

```bash
python src/preprocess.py
```

---

### 2️⃣ `tokenizer.py`

**Purpose:** Tokenization and vocabulary generation.

This script:

* Tokenizes English and Hindi sentences
* Creates vocabularies
* Assigns numerical IDs/indexes to words
* Converts textual data into a format suitable for the HMM

**Run second:**

```bash
python src/tokenizer.py
```

---

### 3️⃣ `train_hmm.py`

**Purpose:** Initial HMM model training.

This script calculates the initial model parameters, including:

* **Transition probabilities** — probability of moving from one state/word to another
* **Emission probabilities** — probability of observing a Hindi word given an English word/state
* **Initial probabilities** — probability of starting with a particular state

**Run third:**

```bash
python src/train_hmm.py
```

---

### 4️⃣ `baum_welch.py`

**Purpose:** HMM parameter optimization.

This script applies the **Baum-Welch algorithm**, an Expectation-Maximization (EM) algorithm, to refine the HMM parameters.

It optimizes the model's probability distributions so that the learned model can produce better translation predictions.

**Run fourth:**

```bash
python src/baum_welch.py
```

---

# 🔮 Prediction / Inference Phase

After training is complete, the trained model is used to translate new English sentences.

### 5️⃣ `viterbi.py`

**Purpose:** Hindi word-sequence prediction using the **Viterbi decoding algorithm**.

The Viterbi algorithm finds the most probable sequence of Hindi states/words for a given English input sequence.

**Run fifth:**

```bash
python src/viterbi.py
```

Example:

```text
Enter English sentence:
i am going

Hindi Translation:
मैं जा रहा हूँ
```

---

### 6️⃣ `translator.py`

**Purpose:** Continuous translation engine.

This script acts as the main command-line translation interface. It loads the trained HMM model and repeatedly accepts English sentences from the user, generating Hindi translations.

**Run sixth:**

```bash
python src/translator.py
```

This starts a continuous translation loop where users can enter multiple sentences without restarting the program.

---

### 7️⃣ `app.py`

**Purpose:** Final web application.

The Streamlit application provides a user-friendly interface for interacting with the trained translation model through a web browser.

**Run finally:**

```bash
streamlit run app.py
```

Streamlit launches the application in your browser, providing an interactive English-to-Hindi translation interface. 🚀

---

## 📌 Execution Order

Always execute the scripts in the following order:

```text
1. preprocess.py
       ↓
2. tokenizer.py
       ↓
3. train_hmm.py
       ↓
4. baum_welch.py
       ↓
5. viterbi.py
       ↓
6. translator.py
       ↓
7. app.py
```

### In simple terms:

**Prepare the data → Tokenize it → Train the HMM → Optimize the model → Decode translations → Run the translation engine → Launch the web application.**

This separation makes the project easier to understand by clearly distinguishing the **model-building/training pipeline** from the **translation/inference pipeline**.

