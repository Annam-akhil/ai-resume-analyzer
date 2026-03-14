import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🚀 AI Resume Analyzer & ATS Optimizer")

# ----------------------------------
# LOAD MODELS
# ----------------------------------

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    suggestion_model = pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

    return embedding_model, suggestion_model


embedding_model, suggestion_model = load_models()

# ----------------------------------
# SAMPLE JOB DESCRIPTIONS
# ----------------------------------

job_roles = {
    "Data Scientist":
        "Python machine learning deep learning NLP pandas numpy scikit-learn data analysis",

    "Machine Learning Engineer":
        "Python machine learning deep learning pytorch tensorflow model deployment docker",

    "AI Engineer":
        "Python NLP transformers huggingface deep learning LLMs AI systems",

    "Backend Developer":
        "Python Flask API development SQL docker cloud backend systems",

    "Frontend Developer":
        "React javascript html css frontend web development"
}

# Generate job embeddings dynamically
jd_embeddings = {}

for job, description in job_roles.items():
    jd_embeddings[job] = embedding_model.encode(description)

# ----------------------------------
# SKILLS LIST
# ----------------------------------

skills_list = [
    "python","machine learning","deep learning","nlp","pytorch","tensorflow",
    "scikit-learn","sql","docker","kubernetes","aws","azure","gcp",
    "data science","data analysis","react","javascript","html","css",
    "flask","streamlit","pandas","numpy","hugging face","transformers"
]

# ----------------------------------
# PDF TEXT EXTRACTION
# ----------------------------------

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text.lower()

# ----------------------------------
# SKILL EXTRACTION
# ----------------------------------

def extract_skills(text):

    found = []

    for skill in skills_list:

        if skill in text:
            found.append(skill)

    return found

# ----------------------------------
# FILE UPLOAD
# ----------------------------------

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:

    st.success("Resume Uploaded Successfully")

    resume_text = extract_text(uploaded_file)

    st.subheader("📄 Extracted Resume Text")
    st.write(resume_text[:1200])

    # ----------------------------------
    # CREATE EMBEDDING
    # ----------------------------------

    resume_embedding = embedding_model.encode([resume_text])[0]

    # ----------------------------------
    # JOB DESCRIPTION MATCH
    # ----------------------------------

    st.subheader("📋 Paste Job Description")

    job_description = st.text_area("Paste the Job Description here")

    if job_description:

        jd_embedding = embedding_model.encode([job_description])[0]

        jd_score = cosine_similarity(
            np.array(resume_embedding).reshape(1,-1),
            np.array(jd_embedding).reshape(1,-1)
        )[0][0]

        jd_score = round(jd_score * 100, 2)

        st.subheader("📊 Resume vs Job Description Match")

        if jd_score < 50:
            st.error(f"Match Score: {jd_score}% (Low Match)")
        elif jd_score < 70:
            st.warning(f"Match Score: {jd_score}% (Moderate Match)")
        else:
            st.success(f"Match Score: {jd_score}% (Strong Match)")

        st.progress(float(jd_score) / 100)

    # ----------------------------------
    # JOB MATCHING
    # ----------------------------------

    scores = []

    for job_name, jd_emb in jd_embeddings.items():

        sim = cosine_similarity(
            np.array(resume_embedding).reshape(1,-1),
            np.array(jd_emb).reshape(1,-1)
        )[0][0]

        scores.append([job_name, sim])

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # ----------------------------------
    # ATS SCORE
    # ----------------------------------

    best_match_score = scores[0][1]

    ats_score = round(best_match_score * 100, 2)

    st.subheader("📊 ATS Resume Score")

    if ats_score < 50:
        st.error(f"ATS Score: {ats_score}% (Needs Improvement)")
    elif ats_score < 70:
        st.warning(f"ATS Score: {ats_score}% (Moderate)")
    else:
        st.success(f"ATS Score: {ats_score}% (Strong Resume)")

    # ----------------------------------
    # ATS GAUGE
    # ----------------------------------

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ats_score,
        title={'text': "ATS Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "lightgreen"}
            ]
        }
    ))

    st.plotly_chart(fig)

    # ----------------------------------
    # SKILLS DETECTION
    # ----------------------------------

    st.subheader("🧠 Detected Skills")

    detected_skills = extract_skills(resume_text)

    if detected_skills:
        st.write(detected_skills)
    else:
        st.warning("No major skills detected")

    # ----------------------------------
    # SKILL RADAR CHART
    # ----------------------------------

    st.subheader("📡 Skill Radar Chart")

    skill_scores = []

    for skill in skills_list:

        if skill in detected_skills:
            skill_scores.append(1)
        else:
            skill_scores.append(0)

    radar_df = pd.DataFrame({
        "Skill": skills_list,
        "Presence": skill_scores
    })

    fig = px.line_polar(
        radar_df,
        r="Presence",
        theta="Skill",
        line_close=True
    )

    fig.update_traces(fill='toself')

    st.plotly_chart(fig)

    # ----------------------------------
    # MISSING SKILLS
    # ----------------------------------

    missing_skills = [s for s in skills_list if s not in detected_skills]

    st.subheader("❗ Missing Skills")

    st.write(missing_skills[:10])

    # ----------------------------------
    # TOP MATCHING JOBS
    # ----------------------------------

    st.subheader("🎯 Top Matching Jobs")

    df = pd.DataFrame(scores[:5], columns=["Job Role","Match Score"])

    df["Match Score"] = (df["Match Score"] * 100).round(2)

    st.dataframe(df)

    # ----------------------------------
    # AI SUGGESTIONS
    # ----------------------------------

    st.subheader("🤖 AI Resume Suggestions")

    prompt = f"""
    Analyze the following resume and provide suggestions to improve ATS score.

    Resume:
    {resume_text}
    """

    suggestions = suggestion_model(prompt, max_length=200)

    st.write(suggestions[0]["generated_text"])
