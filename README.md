🧠 Mercor AI Text Detection — DeBERTa & RoBERTa Ensemble

🏆 Top 100 Globally — Mercor AI Text Detection Challenge

📘 Overview

This project was developed for the Mercor AI Text Detection Challenge, a global competition focused on classifying whether a written response is AI-generated or human-written.
Out of thousands of participants worldwide, this solution ranked in the Top 100 globally, highlighting its strong performance and robust ensemble design.

We fine-tune two transformer-based models — DeBERTa-v3-small and RoBERTa-base — and combine their predictions using a weighted ensemble to achieve a balanced and accurate final classifier.

⚙️ Environment

Frameworks & Libraries

🧩 PyTorch 2.6.0 + cu124 — model training & GPU acceleration

🤗 Transformers 4.53.3 — pre-trained transformer architectures

📦 Datasets 4.4.1 — efficient text loading and tokenization

🧪 scikit-learn, pandas, numpy — preprocessing and evaluation

📂 Dataset

Path: /kaggle/input/mercor-ai-detection

File	Description
train.csv	Training data with is_cheating labels
test.csv	Unlabeled test data
sample_submission.csv	Submission template

Columns

Column	Description
id	Unique identifier
topic	Question or prompt
answer	Written response
is_cheating	Target (1 = AI-generated, 0 = human-written)
🧩 Model Architecture
1. DeBERTa-v3-small

Disentangled attention, superior contextual understanding for nuanced text classification.

2. RoBERTa-base

Optimized BERT variant emphasizing pre-training robustness and text fluency.

3. Ensemble

🤝 Why Use an Ensemble?¶
An ensemble combines predictions from multiple models to make a single, more stable prediction.
Instead of relying on one model’s biases or weaknesses, the ensemble leverages their strengths — like averaging multiple expert opinions.

In this notebook: ```python final_preds = (deberta_preds + roberta_preds) / 2
	​
Weighted averaging yields smoother decision boundaries and improved F1 balance.

🚀 Training Pipeline

Load and preprocess data (tokenization, truncation).

Fine-tune both transformer models on is_cheating.

Evaluate using F1 Score.

Save model checkpoints and inference outputs.

Blend predictions via ensemble weighting for submission.

📈 Evaluation Metric

F1 Score — harmonic mean of precision and recall

F1=
precision+recall
2×(precision×recall)
	​


Chosen to reward balanced accuracy across both AI-generated and human-written classes.
