# %% [markdown]
# # MAP: Misconception Annotation Project
# This notebook demonstrates an end-to-end workflow for the **MAP** Kaggle competition. It covers data exploration, feature engineering, baseline and transformer models, ensembling, and submission generation. The goal is to predict misconceptions from student answers and evaluate using the MAP@3 metric.

# %% [markdown]
# ## Environment Setup
# Install the necessary packages. If running locally you may remove the exclamation mark. In this example the cell is ready for use in a Jupyter environment.

# %%
import subprocess
import sys

# Install required packages when running outside of Jupyter.
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "transformers",
    "datasets",
    "scikit-learn",
    "pandas",
    "numpy",
    "torch",
    "tqdm",
])

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from scipy.sparse import hstack
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import unittest
from tqdm.auto import tqdm

# %% [markdown]
# ## Load Data
# Normally you would load `train.csv`, `validation.csv`, and `test.csv` provided by the competition. Here we create small example files so the notebook runs anywhere.

# %%
example_train = pd.DataFrame({
    'row_id': range(1, 6),
    'text': [
        'Physics concept about force 1',
        'Chemistry answer about atoms 2',
        'Biology cells explanation 3',
        'Physics energy statement 4.5',
        'Chemistry reactions note 5'
    ],
    'Category': ['Physics', 'Chemistry', 'Biology', 'Physics', 'Chemistry'],
    'Misconception': ['Force', 'Atoms', 'Cells', 'Energy', 'Reactions']
})
example_valid = pd.DataFrame({
    'row_id': range(6, 8),
    'text': ['Physics mass comment 6', 'Biology DNA 7.2'],
    'Category': ['Physics', 'Biology'],
    'Misconception': ['Mass', 'DNA']
})
example_test = pd.DataFrame({
    'row_id': range(8, 10),
    'text': ['Chemistry bond question 8', 'Physics velocity 9.1']
})
example_train.to_csv('train.csv', index=False)
example_valid.to_csv('validation.csv', index=False)
example_test.to_csv('test.csv', index=False)

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test.csv')

print(train_df.head())
print(val_df.describe(include='all'))

# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# We check label distribution and text characteristics.

# %%
print('Category counts:\n', train_df['Category'].value_counts())
print('\nTop misconceptions:\n', train_df['Misconception'].value_counts())
print('\nExample responses:')
print(train_df['text'].head())
train_df['text_length'] = train_df['text'].str.len()
train_df['text_length'].hist()

# %% [markdown]
# ## Preprocessing & Feature Engineering
# We lowercase text, keep digits and decimal points, collapse whitespace, and count numeric values.

# %%
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\. ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_numeric_counts(text: str) -> dict:
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return {'digit_count': len(numbers)}

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['clean_text'] = df['text'].apply(clean_text)
    df['digit_count'] = df['text'].apply(lambda x: extract_numeric_counts(x)['digit_count'])
    return df

train_df = preprocess_dataframe(train_df)
val_df = preprocess_dataframe(val_df)

# %% [markdown]
# ## Baseline Model: TF–IDF + Logistic Regression
# We fit word and character TF–IDF features then train a logistic regression classifier.

# %%
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['Category'] + ':' + train_df['Misconception'])
val_labels = label_encoder.transform(val_df['Category'] + ':' + val_df['Misconception'])

word_vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5))

X_train_word = word_vectorizer.fit_transform(train_df['clean_text'])
X_train_char = char_vectorizer.fit_transform(train_df['clean_text'])
X_train = hstack([X_train_word, X_train_char])

X_val_word = word_vectorizer.transform(val_df['clean_text'])
X_val_char = char_vectorizer.transform(val_df['clean_text'])
X_val = hstack([X_val_word, X_val_char])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)

val_pred_proba = clf.predict_proba(X_val)

# %%
def map3_score(y_true, y_pred_probs) -> float:
    top = np.argsort(-y_pred_probs, axis=1)[:, :3]
    score = 0.0
    for true, preds in zip(y_true, top):
        for i, p in enumerate(preds, start=1):
            if p == true:
                score += 1.0 / i
                break
    return score / len(y_true)

# %%
class TestMap3Score(unittest.TestCase):
    def test_basic(self):
        y_true = np.array([0, 1])
        preds = np.array([[0.1, 0.9], [0.8, 0.2]])
        self.assertAlmostEqual(map3_score(y_true, preds), 0.5)

    def test_perfect(self):
        y_true = np.array([1, 0])
        preds = np.array([[0.1, 0.9], [0.6, 0.4]])
        self.assertAlmostEqual(map3_score(y_true, preds), 1.0)

suite = unittest.TestLoader().loadTestsFromTestCase(TestMap3Score)
unittest.TextTestRunner(verbosity=2).run(suite)

# %%
baseline_map3 = map3_score(val_labels, val_pred_proba)
print('Baseline validation MAP@3:', baseline_map3)

# %% [markdown]
# ## Transformer Fine-Tuning
# We fine-tune a small HuggingFace transformer model on the training data.

# %%
model_name = 'distilroberta-base'
num_labels = len(label_encoder.classes_)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

train_dataset = Dataset.from_pandas(train_df[['clean_text']])
train_dataset = train_dataset.map(lambda x: tokenizer(x['clean_text'], truncation=True), batched=True)
train_dataset = train_dataset.add_column('labels', train_labels)

val_dataset = Dataset.from_pandas(val_df[['clean_text']])
val_dataset = val_dataset.map(lambda x: tokenizer(x['clean_text'], truncation=True), batched=True)
val_dataset = val_dataset.add_column('labels', val_labels)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs'
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    return {'map@3': map3_score(labels, probs)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
transformer_eval = trainer.predict(val_dataset)
transformer_map3 = transformer_eval.metrics['test_map@3']
print('Transformer validation MAP@3:', transformer_map3)

# %% [markdown]
# ## Ensembling
# We average probabilities from the baseline and transformer models.

# %%
blend_transformer_probs = torch.nn.functional.softmax(
    torch.tensor(transformer_eval.predictions), dim=1
).numpy()
blend_probs = (val_pred_proba + blend_transformer_probs) / 2
ensemble_map3 = map3_score(val_labels, blend_probs)
print('Ensembled validation MAP@3:', ensemble_map3)

# %% [markdown]
# ## Final Predictions & Submission
# Create predictions for the test set and format them for submission.

# %%
test_df = preprocess_dataframe(test_df)
X_test_word = word_vectorizer.transform(test_df['clean_text'])
X_test_char = char_vectorizer.transform(test_df['clean_text'])
X_test = hstack([X_test_word, X_test_char])
base_test_probs = clf.predict_proba(X_test)

hf_test = Dataset.from_pandas(test_df[['clean_text']])
hf_test = hf_test.map(lambda x: tokenizer(x['clean_text'], truncation=True), batched=True)
transformer_test = trainer.predict(hf_test)
trans_test_probs = torch.nn.functional.softmax(
    torch.tensor(transformer_test.predictions), dim=1
).numpy()

test_probs = (base_test_probs + trans_test_probs) / 2

pred_indices = np.argsort(-test_probs, axis=1)[:, :3]
pred_labels = label_encoder.inverse_transform(pred_indices.flatten()).reshape(pred_indices.shape)
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': [' '.join(row) for row in pred_labels]
})
print(submission)
submission.to_csv('submission.csv', index=False)

# %% [markdown]
# ## Conclusion & Next Steps
# This notebook presented a full workflow for the MAP competition with synthetic data. In practice you would train on the full dataset, tune hyperparameters, and consider more advanced ensembling strategies or data augmentation to further improve performance.

