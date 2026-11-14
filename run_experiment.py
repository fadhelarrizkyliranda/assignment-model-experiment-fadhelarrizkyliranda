import os
import sys
import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env if present
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Setup proxy if specified in environment variable
# (e.g. export HTTPS_PROXY="http://...")

from google import genai
from google.genai import types

data_path = os.path.join(BASE_DIR, 'data', 'customer_reviews_sentiment.csv')
df = pd.read_csv(data_path)
print(f'Total data: {len(df)}')
print(df['sentiment'].value_counts())

X = df['review_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(random_state=42)
start_train = time.time()
clf.fit(X_train_vec, y_train)
train_time_classic = time.time() - start_train

start_pred = time.time()
y_pred_classic = clf.predict(X_test_vec)
pred_time_classic = time.time() - start_pred

acc_classic = accuracy_score(y_test, y_pred_classic)
prec_classic = precision_score(y_test, y_pred_classic, pos_label='positif')
rec_classic = recall_score(y_test, y_pred_classic, pos_label='positif')
f1_classic = f1_score(y_test, y_pred_classic, pos_label='positif')
cm_classic = confusion_matrix(y_test, y_pred_classic, labels=['negatif', 'positif'])

print('=== Classical Model Results ===')
print(f'Accuracy: {acc_classic:.4f}')
print(f'Precision: {prec_classic:.4f}')
print(f'Recall: {rec_classic:.4f}')
print(f'F1-Score: {f1_classic:.4f}')
print('Confusion Matrix (negatif, positif):')
print(cm_classic)

api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
if not api_key:
    print('Warning: GEMINI_API_KEY not set. Check .env or system environment.')

client = genai.Client(api_key=api_key, http_options={'timeout': 60000})

def classify_sentiment_llm(review_text):
    prompt = (
        'Klasifikasikan sentimen ulasan berikut sebagai "positif" atau "negatif".\n'
        'Jawab hanya dengan satu kata: positif atau negatif.\n\n'
        f'Ulasan: "{review_text}"\n'
        'Sentimen:'
    )
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=10
                )
            )
            res = response.text.strip().lower()
            if 'positif' in res:
                return 'positif'
            elif 'negatif' in res:
                return 'negatif'
            return res
        except Exception as e:
            print(f'Attempt {attempt+1} failed ({e}). Retrying in 2s...', flush=True)
            time.sleep(2)
    return 'positif'

print('=== Running LLM Inference on Test Set (40 samples) ===')
y_pred_llm = []
start_llm = time.time()
for idx, (i, review) in enumerate(X_test.items()):
    pred = classify_sentiment_llm(review)
    y_pred_llm.append(pred)
    if (idx + 1) % 10 == 0 or idx == len(X_test) - 1:
        print(f'Processed {idx+1}/{len(X_test)}...')
    time.sleep(0.05)
total_time_llm = time.time() - start_llm

y_pred_llm = np.array(y_pred_llm)

acc_llm = accuracy_score(y_test, y_pred_llm)
prec_llm = precision_score(y_test, y_pred_llm, pos_label='positif')
rec_llm = recall_score(y_test, y_pred_llm, pos_label='positif')
f1_llm = f1_score(y_test, y_pred_llm, pos_label='positif')
cm_llm = confusion_matrix(y_test, y_pred_llm, labels=['negatif', 'positif'])

print('=== LLM API Results ===')
print(f'Accuracy: {acc_llm:.4f}')
print(f'Precision: {prec_llm:.4f}')
print(f'Recall: {rec_llm:.4f}')
print(f'F1-Score: {f1_llm:.4f}')
print('Confusion Matrix (negatif, positif):')
print(cm_llm)
print(f'Total inference time: {total_time_llm:.2f}s (avg: {total_time_llm/len(X_test):.2f}s/req)')

results = {
    'acc_classic': float(acc_classic),
    'prec_classic': float(prec_classic),
    'rec_classic': float(rec_classic),
    'f1_classic': float(f1_classic),
    'cm_classic': cm_classic.tolist(),
    'acc_llm': float(acc_llm),
    'prec_llm': float(prec_llm),
    'rec_llm': float(rec_llm),
    'f1_llm': float(f1_llm),
    'cm_llm': cm_llm.tolist(),
    'total_time_llm': float(total_time_llm),
    'pred_time_classic': float(pred_time_classic),
    'train_time_classic': float(train_time_classic),
    'y_test': list(y_test),
    'y_pred_classic': list(y_pred_classic),
    'y_pred_llm': list(y_pred_llm),
    'test_indices': list(X_test.index)
}
with open(os.path.join(BASE_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print('Results successfully saved to results.json!')
