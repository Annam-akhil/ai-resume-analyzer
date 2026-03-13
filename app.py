import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader
import pickle
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
# LOAD JD EMBEDDINGS
# ----------------------------------

with open("output/jd_embeddings_large.pkl", "rb") as f:
    jd_embeddings = pickle.load(f)

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
    # SKILL ANALYSIS CHART
    # ----------------------------------

    st.subheader("📊 Skill Analysis Chart")

    skill_data = {
        "Category":["Detected Skills","Missing Skills"],
        "Count":[len(detected_skills), len(missing_skills)]
    }

    chart_df = pd.DataFrame(skill_data)

    st.bar_chart(chart_df.set_index("Category"))

    # ----------------------------------
    # TOP MATCHING JOBS
    # ----------------------------------

    st.subheader("🎯 Top Matching Jobs")

    df = pd.DataFrame(scores[:5], columns=["Job Role","Match Score"])

    df["Match Score"] = (df["Match Score"] * 100).round(2)

    st.dataframe(df)

    # ----------------------------------
    # MATCH VISUALIZATION
    # ----------------------------------

    for job, score in scores[:5]:

        st.write(job)

        st.progress(float(score))

        st.write(f"{round(score*100,2)} % match")

    # ----------------------------------
    # AI SUGGESTIONS
    # ----------------------------------

    st.subheader("🤖 AI Resume Suggestions")

    prompt = f"""
    Analyze the following resume and provide suggestions to improve ATS score.

    Resume:
    {resume_text}

    Provide:
    - Resume improvement suggestions
    - Missing technical skills
    - Formatting improvements
    """

    suggestions = suggestion_model(prompt, max_length=250)

    st.write(suggestions[0]["generated_text"])

    # ----------------------------------
    # AI RESUME REWRITER
    # ----------------------------------

    st.subheader("✍ AI Resume Rewriter")

    if st.button("Improve My Resume"):

        rewrite_prompt = f"""
        Rewrite the following resume to make it more professional,
        ATS optimized and impactful.

        Resume:
        {resume_text}
        """

        improved_resume = suggestion_model(rewrite_prompt, max_length=400)

        st.subheader("🚀 Improved Resume Version")

        st.write(improved_resume[0]["generated_text"])

    # ----------------------------------
    # CHATBOT
    # ----------------------------------

    st.subheader("🤖 Resume AI Assistant")

    user_question = st.text_input("Ask anything about your resume")

    if user_question:

        chat_prompt = f"""
        Resume:
        {resume_text}

        User Question:
        {user_question}

        Provide a helpful answer.
        """

        answer = suggestion_model(chat_prompt, max_length=200)

        st.write(answer[0]["generated_text"])