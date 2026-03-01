import pandas as pd
import numpy as np
import re
import json
import ast
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
nltk_stop_words = set(stopwords.words('english'))
stopwords_to_keep = {'no', 'not', 'non', 'off', 'self'}
custom_stop_words = set(nltk_stop_words) - stopwords_to_keep

def clean_text(text):
    if pd.isna(text) or text == '':
        return ''
    text = text.lower()
    text = re.sub(r'(\d+)-(pc|inch|oz|lb|piece)', r'\1 \2', text)
    text = re.sub(r'[^a-z0-9\s.\-]', ' ', text)
    text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)
    text = ' '.join(text.split())
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in custom_stop_words]
    return ' '.join(tokens)

# Load data
df_items = pd.read_csv('data/meta_Home_and_Kitchen_filtered.csv', low_memory=False)
cat_lists = df_items['category'].dropna().apply(ast.literal_eval)
for i in range(6):
    df_items[f'cat_{i+1}'] = cat_lists.apply(lambda x, idx=i: x[idx] if len(x) > idx else None)

df_items_knd = df_items[df_items['cat_2'] == 'Kitchen & Dining'].copy()
df_items_knd['title_1'] = df_items_knd['title'].fillna('').apply(clean_text)

# Load metadata and global filters
with open('data/master_metadata.json', 'r') as f:
    metadata = json.load(f)
with open('data/global_filters.json', 'r') as f:
    global_filters = json.load(f)
global_filters = sorted(global_filters, key=len, reverse=True)

def remove_global_filters(text, filters):
    if not text:
        return text
    for term in filters:
        pattern = r'\b' + re.escape(term) + r'\b'
        text = re.sub(pattern, '', text)
    text = ' '.join(text.split())
    return text

def extract_features(title_text, cat3_name, meta):
    features = {}
    if not title_text or pd.isna(cat3_name) or cat3_name not in meta:
        return (features, title_text if title_text else '')
    cat_meta = meta[cat3_name]
    for attr_name, attr_info in cat_meta.items():
        if attr_info['type'] == 'dictionary':
            values = sorted(attr_info['values'], key=len, reverse=True)
            matched = []
            for val in values:
                pattern = r'\b' + re.escape(val) + r'\b'
                if re.search(pattern, title_text):
                    matched.append(val)
                    title_text = re.sub(pattern, '', title_text)
            if matched:
                features[attr_name] = matched
        elif attr_info['type'] == 'regex':
            matched = []
            for pat in attr_info['patterns']:
                match = re.search(pat, title_text)
                if match:
                    matched.append(match.group(0))
                    title_text = title_text[:match.start()] + title_text[match.end():]
            if matched:
                features[attr_name] = matched
    title_text = ' '.join(title_text.split())
    return (features, title_text)

# Step A
df_items_knd['title_2'] = df_items_knd['title_1'].apply(lambda text: remove_global_filters(text, global_filters))

# Step B
results = df_items_knd.apply(lambda row: extract_features(row['title_2'], row['cat_3'], metadata), axis=1, result_type='expand')
df_items_knd['extracted_features'] = results[0]
df_items_knd['title_2'] = results[1]

# Verification
print(f"Rows with extracted features: {(df_items_knd['extracted_features'].apply(len) > 0).sum()} / {len(df_items_knd)}")
print()

for cat3 in sorted(df_items_knd['cat_3'].dropna().unique()):
    if cat3 not in metadata:
        continue
    sample = df_items_knd[
        (df_items_knd['cat_3'] == cat3) &
        (df_items_knd['extracted_features'].apply(len) > 0)
    ].head(3)
    if len(sample) == 0:
        continue
    print(f"--- {cat3} ---")
    for _, row in sample.iterrows():
        print(f"  title_1: {row['title_1'][:80]}")
        print(f"  title_2: {row['title_2'][:80]}")
        print(f"  features: {row['extracted_features']}")
        print()
