# ============================================================
# INTELLIGENT FREELANCE JOB MATCHER
# Phase 3 — Streamlit Web Application
# Author: Muhammad Abdul Ahad Hashmi
# Reference: CRISP-DM Stage 6 - Deployment
# Paper improvement: Full interactive UI — paper had none!
# ============================================================

import streamlit as st
import json
import joblib
import re
import string
import os
import time
from dotenv import load_dotenv
from google import genai

# ============================================================
# PAGE CONFIG — must be first Streamlit command
# ============================================================
st.set_page_config(
    page_title="Freelance Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — professional dark theme UI
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .match-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #ffffff15;
        margin-bottom: 1rem;
    }
    .match-score {
        font-size: 4rem;
        font-weight: 900;
        line-height: 1;
    }
    .score-high  { color: #00e676; }
    .score-mid   { color: #ffeb3b; }
    .score-low   { color: #ff5252; }
    .match-label {
        font-size: 1rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    .score-label-tag {
        font-size: 0.75rem;
        color: #888;
        margin-bottom: 0.3rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .skill-matched {
        display: inline-block;
        background: #00e67620;
        color: #00e676;
        border: 1px solid #00e67640;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .skill-missing {
        display: inline-block;
        background: #ff525220;
        color: #ff5252;
        border: 1px solid #ff525240;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #fff;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #7b2ff7;
    }
    .info-box {
        background: #1a1a2e;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ccc;
        font-size: 0.9rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS & CONFIG (cached so they load only once)
# ============================================================
@st.cache_resource
def load_models():
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    rf    = joblib.load('models/rf_model.pkl')
    return tfidf, rf

@st.cache_resource
def load_resume():
    with open('resume_context.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_gemini():
    load_dotenv('.env')
    api_key = os.getenv('GEMINI_API_KEY')
    client  = genai.Client(api_key=api_key)
    return client

tfidf, rf     = load_models()
resume        = load_resume()
gemini_client = load_gemini()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_match(job_description):
    cleaned = clean_text(job_description)
    vector  = tfidf.transform([cleaned])
    prob    = rf.predict_proba(vector)[0][1]
    return round(float(prob), 4)

def analyze_skill_gap(job_description):
    your_skills_str = ", ".join(resume['core_skills'])
    prompt = f"""
You are a technical recruiter analyzing a job posting.

JOB DESCRIPTION:
{job_description}

CANDIDATE SKILLS:
{your_skills_str}

Extract required skills and classify as MATCHED or MISSING.
Respond ONLY in this exact JSON format:
{{
    "required_skills": ["skill1", "skill2"],
    "matched_skills": ["skill1"],
    "missing_skills": ["skill2"],
    "match_percentage": 75
}}
"""
    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model='models/gemini-3.1-flash-lite-preview',
                contents=prompt
            )
            raw = response.text.strip()
            raw = re.sub(r'```json|```', '', raw).strip()
            return json.loads(raw)
        except Exception as e:
            if '429' in str(e):
                time.sleep(25 * (attempt + 1))
            else:
                break
    return {"required_skills": [], "matched_skills": [],
            "missing_skills": [], "match_percentage": 0}

def generate_proposal(job_description, match_probability, skill_gap):
    matched  = ", ".join(skill_gap.get('matched_skills', []))
    missing  = ", ".join(skill_gap.get('missing_skills', [])) or "None"
    flagship = resume['flagship_project']
    education = resume['education']
    edu_str = (
        f"{education.get('degree','')} from {education.get('institution','')} "
        f"(CGPA: {education.get('cgpa','')})"
        if isinstance(education, dict) else str(education)
    )
    work = resume['work_experience']
    exp_summary = f"{work[0]['role']} at {work[0]['company']}"
    if len(work) > 1:
        exp_summary += f", {work[1]['role']} at {work[1]['company']}"

    prompt = f"""
You are an expert Upwork freelance copywriter writing for {resume['name']}.

CANDIDATE PROFILE:
- Title: {resume['title']}
- Experience: {resume['experience_years']} years
- Skills: {", ".join(resume['core_skills'][:12])}
- Flagship Project: {flagship['name']}: {flagship['description']}
- Other Projects: {', '.join([p['name'] for p in resume.get('other_projects', [])[:2]])}
- Work: {exp_summary}
- Education: {edu_str}
- Certifications: {', '.join(resume.get('certifications', [])[:2])}
- Tone: {resume['tone']}

JOB DESCRIPTION:
{job_description}

MATCHED SKILLS: {matched}
MISSING SKILLS: {missing}
MATCH SCORE: {match_probability*100:.0f}%

Write a 3-paragraph Upwork cover letter:
- Para 1: Hook — address client's specific problem directly
- Para 2: Proof — reference specific past experience matching this job
- Para 3: CTA — offer a quick call or demo

Rules:
- NO generic openers like "I am writing to apply"
- Mention specific technologies from the job
- Under 250 words
- Confident, human, results-focused
"""
    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model='models/gemini-3.1-flash-lite-preview',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if '429' in str(e):
                time.sleep(25 * (attempt + 1))
            else:
                break
    return "Proposal generation failed. Please try again in 1 minute."

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 👤 Your Profile")
    st.markdown(f"**{resume['name']}**")
    st.markdown(f"*{resume['title']}*")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    match_threshold = st.slider(
        "Match threshold for proposal",
        min_value=40,
        max_value=90,
        value=60,
        step=5,
        help="Proposal generated only if COMBINED score exceeds this"
    )
    st.markdown("---")
    st.markdown("### 📊 How it works")
    st.markdown("""
    1. 📋 Paste a job description
    2. 🤖 ML model predicts category match
    3. 🔍 AI extracts skill gaps
    4. 🧮 Combined score calculated
    5. ✍️ Gemini writes your proposal
    6. 📋 Copy and send on Upwork!
    """)
    st.markdown("---")
    st.markdown("### 🧮 Score Formula")
    st.markdown("""   Combined =
  ML Score  × 40%
+ Skill Gap × 60%Skill match weighted higher
    because specific tools matter
    more than job category.
    """)
    st.markdown("---")
    st.markdown(
        "<div style='color:#555;font-size:0.75rem;text-align:center'>"
        "Built with Scikit-Learn + Gemini AI<br>"
        "Beats research paper KNN (0.80) with RF (0.94) F1"
        "</div>",
        unsafe_allow_html=True
    )

# ============================================================
# MAIN UI
# ============================================================
st.markdown('<div class="main-title">🎯 Freelance Job Matcher</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Paste any Upwork job → '
    'Get realistic match score + tailored proposal instantly</div>',
    unsafe_allow_html=True
)

# ── Input area ─────────────────────────────────────────────
job_input = st.text_area(
    "📋 Paste Upwork job description here:",
    height=200,
    placeholder="Paste the full job description from Upwork...\n\n"
                "Example: We are looking for a Python developer with "
                "machine learning experience..."
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    analyze_btn = st.button(
        "🚀 Analyze Job",
        use_container_width=True,
        type="primary"
    )

st.markdown("---")

# ============================================================
# ANALYSIS PIPELINE
# ============================================================
if analyze_btn:
    if not job_input.strip():
        st.warning("⚠️ Please paste a job description first!")
    else:

        # ── Step 1: ML Prediction (instant) ───────────────
        with st.spinner("🤖 Analyzing job category match..."):
            match_prob = predict_match(job_input)
            match_pct  = match_prob * 100

        # ── ML score verdict ───────────────────────────────
        if match_pct >= 70:
            score_class   = "score-high"
        elif match_pct >= 50:
            score_class   = "score-mid"
        else:
            score_class   = "score-low"

        # ── Step 2: Skill Gap Analysis ─────────────────────
        with st.spinner("🔍 Extracting required skills..."):
            time.sleep(3)
            skill_gap = analyze_skill_gap(job_input)

        matched_skills  = skill_gap.get('matched_skills', [])
        missing_skills  = skill_gap.get('missing_skills', [])
        required_skills = skill_gap.get('required_skills', [])

        # ── Combined Score Calculation ─────────────────────
        # ML score  → job category match (40% weight)
        # Skill gap → specific tools match (60% weight)
        if len(required_skills) > 0:
            skill_match_ratio = len(matched_skills) / len(required_skills)
        else:
            skill_match_ratio = match_prob  # fallback

        combined_score = (match_prob * 0.4) + (skill_match_ratio * 0.6)
        combined_pct   = combined_score * 100

        # ── Combined verdict ───────────────────────────────
        if combined_pct >= 70:
            combined_class   = "score-high"
            combined_verdict = "🔥 Excellent Match!"
            combined_color   = "#00e676"
        elif combined_pct >= 45:
            combined_class   = "score-mid"
            combined_verdict = "⚡ Partial Match"
            combined_color   = "#ffeb3b"
        else:
            combined_class   = "score-low"
            combined_verdict = "❌ Low Match"
            combined_color   = "#ff5252"

        # ── Display three columns ──────────────────────────
        col_ml, col_combined, col_bar = st.columns([1, 1, 2])

        with col_ml:
            st.markdown(f"""
            <div class="match-card">
                <div class="score-label-tag">ML Category Score</div>
                <div class="match-score {score_class}">{match_pct:.0f}%</div>
                <div class="match-label">Job Category Match</div>
            </div>
            """, unsafe_allow_html=True)

        with col_combined:
            st.markdown(f"""
            <div class="match-card">
                <div class="score-label-tag">Combined Score</div>
                <div class="match-score {combined_class}">{combined_pct:.0f}%</div>
                <div class="match-label">Realistic Match</div>
                <div style="color:{combined_color};font-weight:700;
                margin-top:0.5rem">{combined_verdict}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_bar:
            st.markdown(
                '<div class="section-header">📊 Score Breakdown</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"**🤖 ML Category Score** — {match_pct:.0f}% (40% weight)")
            st.progress(match_prob)

            st.markdown(
                f"**🔍 Skill Match Score** — "
                f"{len(matched_skills)}/{len(required_skills) if required_skills else '?'} "
                f"skills ({skill_match_ratio*100:.0f}%) (60% weight)"
            )
            st.progress(float(skill_match_ratio))

            st.markdown(f"""
            <div class="info-box">
                <b>🤖 ML Score:</b> {match_pct:.0f}%
                — job category suits your profile<br>
                <b>🔍 Skill Score:</b>
                {len(matched_skills)}/{len(required_skills) if required_skills else '?'}
                matched ({skill_match_ratio*100:.0f}%)<br>
                <b>🧮 Combined:</b> ({match_pct:.0f}% × 0.4) +
                ({skill_match_ratio*100:.0f}% × 0.6)
                = <b>{combined_pct:.0f}%</b><br><br>
                Proposal threshold: <b>{match_threshold}%</b>
                (based on combined score)
            </div>
            """, unsafe_allow_html=True)

        # ── Skill Gap Display ──────────────────────────────
        st.markdown("---")
        st.markdown(
            '<div class="section-header">🔍 Skill Gap Analysis</div>',
            unsafe_allow_html=True
        )

        col_matched, col_missing = st.columns(2)

        with col_matched:
            st.markdown(f"**✅ Matched Skills ({len(matched_skills)})**")
            if matched_skills:
                badges = "".join([
                    f'<span class="skill-matched">✓ {s}</span>'
                    for s in matched_skills
                ])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.markdown("*No matched skills found*")

        with col_missing:
            st.markdown(f"**❌ Missing Skills ({len(missing_skills)})**")
            if missing_skills:
                badges = "".join([
                    f'<span class="skill-missing">✗ {s}</span>'
                    for s in missing_skills
                ])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.markdown("*No missing skills — perfect match!* 🎉")

        # ── Step 3: Proposal Generation ───────────────────
        st.markdown("---")

        if combined_pct >= match_threshold:
            with st.spinner("✍️ Generating your personalized proposal..."):
                time.sleep(5)
                proposal = generate_proposal(
                    job_input, combined_score, skill_gap
                )

            st.markdown(
                '<div class="section-header">✍️ Your Personalized Proposal</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                "<div class='info-box'>✅ High combined match detected! "
                "Gemini AI has written a tailored proposal based on YOUR "
                "resume and THIS job. Edit it below before copying.</div>",
                unsafe_allow_html=True
            )

            edited_proposal = st.text_area(
                "Edit your proposal before copying:",
                value=proposal,
                height=280,
                label_visibility="collapsed"
            )

            col_dl, col_words = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    label="📥 Download Proposal",
                    data=edited_proposal,
                    file_name="upwork_proposal.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_words:
                word_count = len(edited_proposal.split())
                color = "#00e676" if word_count <= 250 else "#ff5252"
                st.markdown(
                    f"<div style='color:{color};padding-top:0.5rem'>"
                    f"📝 Word count: <b>{word_count}</b>/250</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"""
            <div class="info-box">
                ⚠️ Combined score <b>{combined_pct:.0f}%</b> is below
                your threshold of <b>{match_threshold}%</b>.<br>
                ML category score was {match_pct:.0f}% but skill match
                was only {skill_match_ratio*100:.0f}%
                ({len(matched_skills)}/
                {len(required_skills) if required_skills else '?'} skills).<br><br>
                Lower the threshold in the sidebar or look for a
                better matching job.
            </div>
            """, unsafe_allow_html=True)

# ── Empty state ────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#444">
        <div style="font-size:4rem">🎯</div>
        <div style="font-size:1.2rem;margin-top:1rem">
            Paste a job description above and click
            <b style="color:#7b2ff7">Analyze Job</b>
        </div>
        <div style="font-size:0.9rem;margin-top:0.5rem;color:#333">
            Powered by Random Forest ML + Gemini AI
        </div>
    </div>
    """, unsafe_allow_html=True)