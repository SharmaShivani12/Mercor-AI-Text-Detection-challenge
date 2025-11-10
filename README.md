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

Combines logits from both models:

𝑃
𝑓
𝑖
𝑛
𝑎
𝑙
=
𝛼
⋅
𝑃
𝐷
𝑒
𝐵
𝐸
𝑅
𝑇
𝑎
+
(
1
−
𝛼
)
⋅
𝑃
𝑅
𝑜
𝐵
𝐸
𝑅
𝑇
𝑎
P
final
	​

=α⋅P
DeBERTa
	​

+(1−α)⋅P
RoBERTa
	​


Weighted averaging yields smoother decision boundaries and improved F1 balance.

🚀 Training Pipeline

Load and preprocess data (tokenization, truncation).

Fine-tune both transformer models on is_cheating.

Evaluate using F1 Score.

Save model checkpoints and inference outputs.

Blend predictions via ensemble weighting for submission.

📈 Evaluation Metric

F1 Score — harmonic mean of precision and recall:

𝐹
1
=
2
×
(
𝑝
𝑟
𝑒
𝑐
𝑖
𝑠
𝑖
𝑜
𝑛
×
𝑟
𝑒
𝑐
𝑎
𝑙
𝑙
)
𝑝
𝑟
𝑒
𝑐
𝑖
𝑠
𝑖
𝑜
𝑛
+
𝑟
𝑒
𝑐
𝑎
𝑙
𝑙
F1=
precision+recall
2×(precision×recall)
	​


Chosen to reward balanced accuracy across both AI-generated and human-written classes.
