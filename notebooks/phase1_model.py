# ============================================================
# INTELLIGENT FREELANCE JOB MATCHER
# Phase 1 - Steps 3 to 7 (Complete Pipeline)
# Dataset: Upwork Jobs Sample (50,000 real jobs)
# Reference: CRISP-DM Methodology - Sahnoun & Elhadjamor (2024)
# ============================================================

import pandas as pd
import numpy as np
import json
import re
import string
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 55)
print("  FREELANCE JOB MATCHER — Phase 1 Starting...")
print("=" * 55)

# ============================================================
# STEP 3 — Load & Understand the Dataset
# ============================================================
print("\n[Step 3] Loading dataset...")

with open('../data/jobs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Total jobs loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ── Extract skills from ontologySkills column ──────────────
# Each job has a list of skill objects like:
# [{'prefLabel': 'Python'}, {'prefLabel': 'Flask'}...]
# We extract them into a clean comma-separated string

def extract_skills(skills_list):
    """Extract skill names from the ontologySkills field."""
    if not isinstance(skills_list, list):
        return ""
    return ", ".join([
        s.get('prefLabel', '') 
        for s in skills_list 
        if s.get('prefLabel')
    ])

df['skills_text'] = df['ontologySkills'].apply(extract_skills)

# ── Extract category from nested jobDetailsResponse ───────
def extract_category(job_detail):
    try:
        return job_detail['data']['jobPubDetails']['opening']['category']['name']
    except:
        return "Unknown"

df['category'] = df['jobDetailsResponse'].apply(extract_category)

print(f"\nSample title: {df['title'].iloc[0]}")
print(f"Sample skills: {df['skills_text'].iloc[0]}")
print(f"Sample category: {df['category'].iloc[0]}")
print(f"\nTop 10 categories:")
print(df['category'].value_counts().head(10))

# ============================================================
# STEP 4 — Auto-Labeling (Match = 1, No Match = 0)
# Reference: Binary labeling — Paper Section 3.3.2
# Strategy: combine BOTH description + skills for better accuracy
# This is our improvement over the paper — they had no skills data!
# ============================================================
print("\n[Step 4] Auto-labeling jobs...")

YOUR_SKILLS = [
    # Python & backend
    'python', 'django', 'flask', 'fastapi', 'backend',
    # AI & ML
    'machine learning', 'deep learning', 'neural network',
    'scikit', 'tensorflow', 'keras', 'nlp', 'ai',
    'artificial intelligence', 'data science', 'ml',
    'computer vision', 'llm', 'chatbot', 'generative ai',
    # Data
    'data analysis', 'data mining', 'pandas', 'numpy',
    'data visualization', 'matplotlib', 'powerbi',
    # APIs & Automation
    'api', 'rest api', 'api integration', 'automation',
    'web scraping', 'selenium', 'workflow automation',
    # Web
    'full stack', 'fullstack', 'web development',
    # Deployment
    'railway', 'vercel', 'render', 'docker',
    # Database
    'sql', 'postgresql', 'mysql', 'database',
]

def auto_label(row):
    """
    Improved labeling with NEGATIVE keyword penalty.
    If a job is dominated by non-Python skills → force Match=0
    even if it contains some Python keywords.
    """
    combined = " ".join([
        str(row.get('description', '')),
        str(row.get('title', '')),
        str(row.get('skills_text', ''))
    ]).lower()

    # ── Negative keywords — jobs you do NOT want ──────────
    NEGATIVE_KEYWORDS = [
        # Design
        'photoshop', 'illustrator', 'indesign', 'figma',
        'graphic design', 'logo design', 'brand identity',
        'adobe creative', 'canva', 'after effects',
        'motion graphics', 'typography', 'print design',
        # Pure frontend
        'react', 'angular', 'vue', 'next.js', 'nuxt',
        'tailwind', 'css animation', 'sass', 'scss',
        # WordPress dominated
        'wordpress developer', 'woocommerce', 'elementor',
        'divi', 'wpbakery', 'shopify developer',
        # Video/Audio
        'video editing', 'premiere pro', 'final cut',
        'davinci resolve', 'after effects', 'voiceover',
        # Writing
        'copywriting', 'content writing', 'blog writing',
        'seo writing', 'article writing', 'ghostwriting',
        # Marketing
        'social media manager', 'instagram', 'tiktok',
        'facebook ads', 'google ads', 'email marketing',
        'influencer', 'campaign management',
        # Other non-dev
        'bookkeeping', 'accounting', 'quickbooks',
        'customer support', 'virtual assistant',
        'data entry', 'transcription', 'translation'
    ]

    # ── Positive keywords — jobs you DO want ──────────────
    POSITIVE_KEYWORDS = [
        'python', 'django', 'flask', 'fastapi',
        'machine learning', 'deep learning', 'neural network',
        'scikit', 'tensorflow', 'keras', 'nlp',
        'artificial intelligence', 'data science', 'ml',
        'computer vision', 'llm', 'chatbot',
        'data analysis', 'data mining', 'pandas', 'numpy',
        'api integration', 'rest api', 'automation script',
        'web scraping', 'workflow automation',
        'postgresql', 'mysql', 'mongodb', 'database'
    ]

    # ── Count matches ──────────────────────────────────────
    positive_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in combined)
    negative_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in combined)

    # ── Decision logic ─────────────────────────────────────
    # If more negative hits than positive → No Match
    if negative_hits > positive_hits:
        return 0
    # If at least 2 positive keywords → Match
    if positive_hits >= 2:
        return 1
    # If exactly 1 positive and no negatives → Match
    if positive_hits == 1 and negative_hits == 0:
        return 1

    return 0

df['Match'] = df.apply(auto_label, axis=1)

# ── Check distribution ─────────────────────────────────────
match_counts = df['Match'].value_counts()
match_rate = df['Match'].mean() * 100
print(f"\nMatch=1 (High Match jobs): {match_counts.get(1, 0)}")
print(f"Match=0 (Low Match jobs):  {match_counts.get(0, 0)}")
print(f"Match rate: {match_rate:.1f}%")

# ── Visualize label distribution ───────────────────────────
os.makedirs('../data', exist_ok=True)
plt.figure(figsize=(6, 4))
sns.countplot(x='Match', data=df, palette=['#E8593C', '#1D9E75'])
plt.title('Job Match Distribution')
plt.xticks([0, 1], ['Low Match', 'High Match'])
plt.ylabel('Number of Jobs')
for i, count in enumerate(match_counts.sort_index()):
    plt.text(i, count + 50, str(count), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../data/label_distribution.png', dpi=150)
plt.show()
print("Chart saved → data/label_distribution.png")

# ============================================================
# STEP 5 — Text Preprocessing + TF-IDF Vectorization
# Reference: Paper Section 3.2 - TF-IDF vectorization
# Our improvement: we combine description + title + skills
# into one rich text feature — paper only used description!
# ============================================================
print("\n[Step 5] Preprocessing text + TF-IDF vectorization...")

def clean_text(text):
    """
    Cleans job description text.
    Steps: lowercase → remove numbers → remove punctuation → strip spaces
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Combine all text fields for richer features ───────────
# KEY ADVANTAGE over paper: we use title + description + skills
df['combined_text'] = (
    df['title'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['skills_text'].fillna('')
)
df['clean_text'] = df['combined_text'].apply(clean_text)

print("Sample cleaned text (first 200 chars):")
print(df['clean_text'].iloc[0][:200])

# ── TF-IDF Vectorization ───────────────────────────────────
# ngram_range=(1,2) captures phrases like "machine learning"
# as a single feature — critical for tech skill detection!
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words='english'
)

X = tfidf.fit_transform(df['clean_text'])
y = df['Match']

print(f"\nTF-IDF matrix: {X.shape[0]} jobs × {X.shape[1]} features")

# ── Train-Test Split ───────────────────────────────────────
# stratify=y ensures both splits have same Match% ratio
# Reference: Paper Section 3.4 - Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Training: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

# ============================================================
# STEP 6 — Train ML Models + Evaluate
# Reference: Paper Section 3.3 - KNN got Precision=0.80
# Our target: BEAT 0.80 with Random Forest
# ============================================================
print("\n[Step 6] Training models...")

# ── Model A: Logistic Regression (fast baseline) ───────────
print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ── Model B: Random Forest (main model) ───────────────────
print("  Training Random Forest (this may take ~30 seconds)...")
rf = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

# ── Print results ──────────────────────────────────────────
print("\n── Logistic Regression ──")
print(classification_report(y_test, lr_pred,
      target_names=['Low Match', 'High Match']))

print("── Random Forest ──")
print(classification_report(y_test, rf_pred,
      target_names=['Low Match', 'High Match']))

# ── Compare with paper's results ──────────────────────────
print("\n── Our Results vs Research Paper (KNN) ──")
print(f"{'Metric':<15} {'Paper KNN':>12} {'Our LR':>10} {'Our RF':>10}")
print("-" * 50)
print(f"{'Precision':<15} {'0.80':>12} "
      f"{precision_score(y_test, lr_pred):.3f}{' ':>6}"
      f"{precision_score(y_test, rf_pred):.3f}")
print(f"{'Recall':<15} {'0.60':>12} "
      f"{recall_score(y_test, lr_pred):.3f}{' ':>6}"
      f"{recall_score(y_test, rf_pred):.3f}")
print(f"{'F1 Score':<15} {'0.69':>12} "
      f"{f1_score(y_test, lr_pred):.3f}{' ':>6}"
      f"{f1_score(y_test, rf_pred):.3f}")

# ── Confusion Matrix ───────────────────────────────────────
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Low Match', 'High Match'],
            yticklabels=['Low Match', 'High Match'])
plt.title('Confusion Matrix — Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('../data/confusion_matrix.png', dpi=150)
plt.show()
print("Chart saved → data/confusion_matrix.png")

# ── Quick live test ────────────────────────────────────────
print("\n── Live Test ──")
test_jobs = [
    "Python developer needed for ML model training and API integration using Flask",
    "Looking for a graphic designer with Photoshop and Illustrator skills",
    "Need Django backend developer to build REST API with PostgreSQL database"
]
for job in test_jobs:
    vec = tfidf.transform([clean_text(job)])
    prob = rf.predict_proba(vec)[0][1]
    label = "HIGH MATCH ✅" if prob >= 0.6 else "Low Match ❌"
    print(f"  {prob*100:.0f}% — {label}")
    print(f"  '{job[:60]}...'")

# ============================================================
# STEP 7 — Save Model + Vectorizer with Joblib
# Reference: Paper Section 4 - Deployment preparation
# These files will be loaded instantly by Streamlit in Phase 3
# ============================================================
print("\n[Step 7] Saving models...")

os.makedirs('../models', exist_ok=True)

joblib.dump(tfidf, '../models/tfidf_vectorizer.pkl')
joblib.dump(rf,    '../models/rf_model.pkl')
joblib.dump(lr,    '../models/lr_model.pkl')

print("  Saved → models/tfidf_vectorizer.pkl")
print("  Saved → models/rf_model.pkl")
print("  Saved → models/lr_model.pkl")

# ── Verify they load correctly ─────────────────────────────
loaded_tfidf = joblib.load('../models/tfidf_vectorizer.pkl')
loaded_rf    = joblib.load('../models/rf_model.pkl')

test_vec  = loaded_tfidf.transform(["python machine learning api flask django"])
test_prob = loaded_rf.predict_proba(test_vec)[0][1]
print(f"\n  Verification test: {test_prob*100:.1f}% match confidence")
print("  Models loaded and working correctly ✅")

print("\n" + "=" * 55)
print("  PHASE 1 COMPLETE!")
print("  Your models are saved in the models/ folder.")
print("  Ready to move to Phase 2 — LLM API Integration!")
print("=" * 55)