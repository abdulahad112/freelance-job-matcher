# ============================================================
# INTELLIGENT FREELANCE JOB MATCHER
# Phase 2 — LLM API Integration (Gemini)
# Using updated google-genai package (2025)
# Model: gemini-3.1-flash-lite-preview (500 RPD free tier)
# ============================================================

import os
import json
import joblib
import re
import string
from dotenv import load_dotenv
from google import genai

# ── Load environment variables ─────────────────────────────
load_dotenv('../.env')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found! Check your .env file.")

# ── Configure Gemini ───────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)
print("Gemini API connected ✅")

# ============================================================
# STEP 3A — Load Your Resume Context
# ============================================================
with open('../resume_context.json', 'r', encoding='utf-8') as f:
    resume = json.load(f)

print(f"Resume loaded for: {resume['name']} ✅")

# ============================================================
# STEP 3B — Load Saved ML Model from Phase 1
# ============================================================
tfidf = joblib.load('../models/tfidf_vectorizer.pkl')
rf    = joblib.load('../models/rf_model.pkl')
print("ML models loaded ✅")

# ============================================================
# STEP 4A — Text Cleaning (same as Phase 1)
# ============================================================
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================
# STEP 4B — ML Match Prediction
# ============================================================
def predict_match(job_description):
    """
    Uses our Phase 1 trained Random Forest model to predict
    match probability for a given job description.
    Returns: probability (0.0 to 1.0)
    """
    cleaned = clean_text(job_description)
    vector  = tfidf.transform([cleaned])
    prob    = rf.predict_proba(vector)[0][1]
    return round(prob, 4)

# ============================================================
# STEP 4C — Skill Gap Analyzer
# Reference: Our key improvement over the paper!
# Paper Section 5 mentions "incorporating additional data
# sources could significantly refine recommendations"
# We implement this as a Skill Gap feature using LLM
# ============================================================
def analyze_skill_gap(job_description):
    import time

    your_skills_str = ", ".join(resume['core_skills'])

    prompt = f"""
You are a technical recruiter analyzing a job posting.

JOB DESCRIPTION:
{job_description}

CANDIDATE SKILLS:
{your_skills_str}

Your task:
1. Extract ALL technical skills required in the job description
2. Compare them against the candidate's skills
3. Classify each required skill as either MATCHED or MISSING

Respond ONLY in this exact JSON format, no other text:
{{
    "required_skills": ["skill1", "skill2", "skill3"],
    "matched_skills": ["skill1", "skill2"],
    "missing_skills": ["skill3"],
    "match_percentage": 75
}}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='models/gemini-3.1-flash-lite-preview',
                contents=prompt
            )
            raw = response.text.strip()
            raw = re.sub(r'```json|```', '', raw).strip()
            return json.loads(raw)

        except Exception as e:
            error_str = str(e)
            if '429' in error_str:
                wait = 25 * (attempt + 1)
                print(f"  Rate limit hit. Waiting {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
            else:
                print(f"  API error: {error_str[:100]}")
                break

    return {
        "required_skills": [],
        "matched_skills": [],
        "missing_skills": [],
        "match_percentage": 0
    }

# ============================================================
# STEP 4D — Proposal Generator
# Core improvement over the paper — LLM powered proposal
# writing personalized to YOU
# ============================================================
def generate_proposal(job_description, match_probability, skill_gap):
    """
    Uses Gemini to write a tailored 3-paragraph Upwork proposal.
    Combines: job requirements + your resume + match analysis
    """
    import time

    matched = ", ".join(skill_gap.get('matched_skills', []))
    missing = ", ".join(skill_gap.get('missing_skills', [])) or "None"

    # ── Get flagship project info ──────────────────────────
    flagship = resume['flagship_project']
    flagship_desc = f"{flagship['name']}: {flagship['description']}"

    # ── Get work experience ────────────────────────────────
    work_experience = resume['work_experience']
    exp_summary = f"{work_experience[0]['role']} at {work_experience[0]['company']}"
    if len(work_experience) > 1:
        exp_summary += f", {work_experience[1]['role']} at {work_experience[1]['company']}"

    # ── Get education ──────────────────────────────────────
    education = resume['education']
    if isinstance(education, dict):
        edu_str = f"{education.get('degree', '')} from {education.get('institution', '')} (CGPA: {education.get('cgpa', '')})"
    else:
        edu_str = str(education)

    prompt = f"""
You are an expert Upwork freelance copywriter writing on behalf of {resume['name']}.

CANDIDATE PROFILE:
- Title: {resume['title']}
- Experience: {resume['experience_years']} years
- Core Skills: {", ".join(resume['core_skills'][:12])}
- Flagship Project: {flagship_desc}
- Other Projects: {', '.join([p['name'] for p in resume.get('other_projects', [])[:2]])}
- Work History: {exp_summary}
- Education: {edu_str}
- Certifications: {', '.join(resume.get('certifications', [])[:2])}
- Tone: {resume['tone']}

JOB DESCRIPTION:
{job_description}

MATCHED SKILLS: {matched}
MISSING SKILLS: {missing}
MATCH SCORE: {match_probability*100:.0f}%

Write a professional 3-paragraph Upwork cover letter:
- Paragraph 1: Hook — directly address the client's specific problem/need
- Paragraph 2: Proof — reference 1-2 specific past experiences that match THIS job
- Paragraph 3: CTA — clear next step, offer a quick call or demo

Rules:
- Do NOT use generic phrases like "I am writing to apply"
- DO mention specific technologies from the job description
- Keep it under 250 words
- Sound like a confident human professional, not a robot
- Start with something that grabs attention immediately
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='models/gemini-3.1-flash-lite-preview',
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            error_str = str(e)
            if '429' in error_str:
                wait = 25 * (attempt + 1)
                print(f"  Rate limit hit. Waiting {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
            else:
                print(f"  API error: {error_str[:120]}")
                break

    return "Proposal generation failed after retries. Please try again in 1 minute."

# ============================================================
# STEP 5 — Full Pipeline
# ============================================================
def analyze_job(job_description):
    """
    Master function — runs the complete pipeline:
    1. ML match prediction
    2. Skill gap analysis
    3. Proposal generation (if high match)
    Returns everything Streamlit needs in Phase 3
    """
    import time

    print("\n" + "=" * 55)
    print("Analyzing job...")

    # ── Step 1: ML Prediction (instant, no API) ───────────
    match_prob = predict_match(job_description)
    match_pct  = match_prob * 100
    print(f"Match Score: {match_pct:.1f}%")

    # ── Step 2: Skill Gap Analysis ─────────────────────────
    print("Running skill gap analysis...")
    time.sleep(3)
    skill_gap = analyze_skill_gap(job_description)
    print(f"Matched Skills: {skill_gap.get('matched_skills', [])}")
    print(f"Missing Skills: {skill_gap.get('missing_skills', [])}")

    # ── Step 3: Proposal (only if match >= 60%) ───────────
    proposal = None
    if match_prob >= 0.60:
        print("High match! Generating proposal...")
        time.sleep(5)
        proposal = generate_proposal(job_description, match_prob, skill_gap)
        print("\n── Generated Proposal ──")
        print(proposal)
    else:
        print(f"Low match ({match_pct:.0f}%) — skipping proposal.")

    return {
        "match_probability": match_prob,
        "match_percentage": match_pct,
        "skill_gap": skill_gap,
        "proposal": proposal,
        "is_high_match": match_prob >= 0.60
    }

# ============================================================
# RUN TEST
# ============================================================
if __name__ == "__main__":

    test_job = """
    We are looking for an experienced Python Developer with strong
    knowledge of machine learning and REST API development.

    Requirements:
    - Python (3+ years experience)
    - Flask or Django for backend development
    - Machine learning with scikit-learn or TensorFlow
    - REST API design and integration
    - PostgreSQL or MySQL database experience
    - Experience deploying on cloud platforms (Railway, Heroku, etc.)

    Nice to have:
    - NLP experience
    - Web scraping with BeautifulSoup or Selenium
    - Docker containerization

    You will be building an AI-powered data pipeline that processes
    user inputs and returns intelligent predictions via a REST API.
    This is a long-term project with potential for full-time engagement.
    """

    result = analyze_job(test_job)

    print("\n── FINAL RESULT SUMMARY ──")
    print(f"Match:          {result['match_percentage']:.1f}%")
    print(f"High Match:     {result['is_high_match']}")
    print(f"Matched Skills: {result['skill_gap'].get('matched_skills')}")
    print(f"Missing Skills: {result['skill_gap'].get('missing_skills')}")
    print(f"Proposal ready: {'Yes ✅' if result['proposal'] else 'No (low match)'}")