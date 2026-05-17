if your are new 
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


IMPORTANT UNDERSTANDING
TRAINING PHASE

These files are training phase:

preprocess.py
tokenizer.py
train_hmm.py
baum_welch.py
PREDICTION PHASE

These files are inference phase:

viterbi.py
translator.py
app.py

WHAT EACH FILE DOES
1️⃣ preprocess.py
Purpose
Clean dataset
Remove empty rows
Create cleaned_dataset.csv
Run FIRST
python src/preprocess.py
2️⃣ tokenizer.py
Purpose
Tokenization
Vocabulary creation
Word indexing
Run SECOND
python src/tokenizer.py
3️⃣ train_hmm.py
Purpose

Creates:

transition probabilities
emission probabilities
initial probabilities
Run THIRD
python src/train_hmm.py
4️⃣ baum_welch.py
Purpose

Optimizes HMM probabilities using:

Baum-Welch
EM algorithm
Run FOURTH
python src/baum_welch.py
5️⃣ viterbi.py
Purpose

Runs the:

Viterbi decoding algorithm

Predicts Hindi translation.

Run FIFTH
python src/viterbi.py

Example:

Enter English sentence:
i am going
6️⃣ translator.py
Purpose

Main continuous translator engine.

Run SIXTH
python src/translator.py

This creates live translation loop.

7️⃣ app.py
Purpose

Final Streamlit web application.

Run FINAL
streamlit run app.py

Browser opens automatically 🚀