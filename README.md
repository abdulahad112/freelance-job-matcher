
# 🎯 Intelligent Freelance Job Matcher & Proposal Generator

> **An AI-powered tool that analyzes Upwork job descriptions, predicts match probability using a custom-trained ML model, performs skill gap analysis, and automatically generates tailored proposals using Google Gemini AI.**


## 📸 Screenshots

### Main Interface
<img width="1919" height="907" alt="freelance" src="https://github.com/user-attachments/assets/4d5d5d3b-d56d-4025-889d-930dbd1addd4" />


### Results — Match Score + Skill Gap
<img width="1922" height="2919" alt="screencapture-freelance-job-matcher-4h8h5gra3yvkndl8hjkqzy-streamlit-app-2026-04-20-13_55_02" src="https://github.com/user-attachments/assets/f0c4b87b-cc2e-4ca4-b8c4-8df9e685d24a" />


---

## 🏆 Performance vs Research Paper

This project outperforms the published research paper:
**"Enhanced Freelance Matching: Integrated Data Analysis and Machine Learning Techniques"**
*(Sahnoun & Elhadjamor, Journal of Computing Theories and Applications, 2024)*

| Metric | Paper (KNN) | Our Model (RF) | Improvement |
|--------|------------|----------------|-------------|
| Precision | 0.80 | **0.893** | +11.6% ✅ |
| Recall | 0.60 | **0.723** | +20.5% ✅ |
| F1 Score | 0.69 | **0.799** | +15.8% ✅ |
| LLM Proposals | ❌ None | ✅ Yes | New Feature |
| Skill Gap Analysis | ❌ None | ✅ Yes | New Feature |
| Real Dataset | ❌ Simulated | ✅ 50,000 real jobs | New Feature |
| Web UI | ❌ None | ✅ Streamlit App | New Feature |

---

## ✨ Features

### 🤖 Phase 1 — Custom ML Model
- Trained on **50,000 real Upwork job descriptions** from Kaggle
- **Random Forest classifier** with TF-IDF vectorization
- **Negative keyword penalty system** for accurate filtering
- Bigram features to capture tech phrases like "machine learning"
- Saves trained model with `joblib` for instant loading

### 🧠 Phase 2 — Gemini AI Integration
- **Skill Gap Analyzer** — extracts required skills and compares to your resume
- **Proposal Generator** — writes tailored 3-paragraph cover letters
- Retry logic with rate limit handling
- Personalized using your resume context JSON file

### 🎨 Phase 3 — Streamlit Web App
- **Combined scoring system** — ML score (40%) + Skill match (60%)
- Real-time match probability display
- Color-coded skill badges (matched vs missing)
- Editable proposal text area
- Word count tracker (250 word limit)
- Download proposal as `.txt` file
- Adjustable match threshold slider
- Professional dark theme UI

---

## 🏗️ Project Structure

```
freelance-job-matcher/
├── app.py                    # Streamlit web application (Phase 3)
├── resume_context.json       # Your resume/profile data
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed to git)
├── .gitignore
│
├── notebooks/
│   ├── phase1_model.py       # ML training pipeline (Phase 1)
│   └── phase2_llm.py         # LLM integration testing (Phase 2)
│
├── models/
│   ├── tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer
│   ├── rf_model.pkl          # Saved Random Forest model
│   └── lr_model.pkl          # Saved Logistic Regression model
│
├── data/
│   ├── jobs.json             # Raw Upwork dataset (50k jobs)
│   ├── label_distribution.png
│   └── confusion_matrix.png
│
└── assets/
    └── screenshots/
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))
- Upwork jobs dataset from Kaggle

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/freelance-job-matcher.git
cd freelance-job-matcher

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create .env file in project root:
# GEMINI_API_KEY=your_key_here

# 5. Add your resume to resume_context.json

# 6. Train the ML model
cd notebooks
python phase1_model.py

# 7. Run the app
cd ..
streamlit run app.py
```

---

## 📊 How the Combined Score Works

The tool uses a **two-component scoring system** for realistic predictions:

```
Combined Score = (ML Category Score × 40%) + (Skill Match Score × 60%)
```

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| ML Category Score | 40% | Does this job TYPE suit your profile? |
| Skill Match Score | 60% | Do you have the SPECIFIC tools required? |

**Why 60% weight on skills?** Specific tools matter more than job category on Upwork. A Python developer should not get 98% on a WordPress job just because it mentions "Python scripting."

### Example
```
SaaS + FastAPI job:
  ML Score:       84%  (it's a dev job — correct)
  Skill Score:    30%  (only 3/10 skills matched)
  Combined Score: 52%  ⚡ Partial Match (realistic!)
```

---

## 🗂️ Dataset

- **Source:** Kaggle — "Upwork Jobs Sample of 3.5M Dataset" (50k jobs)
- **Fields used:** `title`, `description`, `ontologySkills`
- **Labeling:** Rule-based auto-labeling with positive + negative keywords
- **Match rate:** ~12.7% (realistic for Python/AI specialization)
- **Train/Test split:** 80/20 with stratification

---

## 🧪 ML Pipeline Details

### Text Preprocessing
1. Lowercase conversion
2. Number removal
3. Punctuation removal
4. Extra whitespace removal
5. Combined: `title + description + skills` → richer features

### TF-IDF Vectorization
```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),    # captures "machine learning" as one feature
    min_df=2,
    stop_words='english'
)
```

### Labeling Strategy
- **Positive keywords:** Python, Django, Flask, ML, AI, FastAPI, etc.
- **Negative keywords:** Photoshop, WordPress, React, social media, etc.
- **Decision logic:** negative_hits > positive_hits → Low Match

### Model
```python
RandomForestClassifier(
    n_estimators=200,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

---

## 🔧 Configuration

### resume_context.json
Customize this file with your own profile:

```json
{
    "name": "Your Name",
    "title": "AI Developer & Python Engineer",
    "experience_years": 3,
    "core_skills": ["Python", "Django", "Flask", "Machine Learning"],
    "flagship_project": {
        "name": "Your Project",
        "description": "What it does and what accuracy it achieved"
    },
    "work_experience": [...],
    "education": {...},
    "tone": "professional but approachable"
}
```

### Adjustable Settings (Streamlit sidebar)
- **Match threshold:** 40–90% (default: 60%)
- Proposal only generated when combined score exceeds threshold

---

## 📚 Academic Reference

This project is based on and improves upon:

> Sahnoun, I., & Elhadjamor, E. A. (2024). Enhanced Freelance Matching: Integrated Data Analysis and Machine Learning Techniques. *Journal of Computing Theories and Applications*, 1(4). DOI: 10.62411/jcta.10152

**Key improvements over the paper:**
1. Real dataset (50k jobs) vs simulated data
2. Negative keyword penalty for realistic filtering
3. Combined scoring (ML + Skill Gap) for accuracy
4. LLM-powered proposal generation
5. Skill gap extraction using Gemini AI
6. Production-ready Streamlit web interface
7. Adjustable threshold with download functionality

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Scikit-Learn (Random Forest + TF-IDF) |
| LLM API | Google Gemini (gemini-3.1-flash-lite) |
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Environment | Python-dotenv |
| Dataset | Kaggle (50k Upwork jobs) |

---

## 👤 Author

**Muhammad Abdul Ahad Hashmi**
- AI Developer & Full Stack Python Engineer
- AI Developer Apprentice — Pakistan Software Export Board (PSEB)
- Former AI Developer — Moboroid Software House
- Former Frontend Developer Intern — Hisky Tech
- BS Information Technology

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

*Built with ❤️ using Python, Scikit-Learn, and Gemini AI*
