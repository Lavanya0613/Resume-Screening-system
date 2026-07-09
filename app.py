import streamlit as st
import os
import html
import re
import os
from dotenv import load_dotenv
from src.resume_parser import extract_text
from src.processor import clean_text, sanitize_for_bias
from src.scorer import calculate_ats_score
from src.llm_feedback import generate_feedback
import pandas as pd

# 1. PAGE CONFIG
st.set_page_config(page_title="AI Resume Screening System", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# Load environment variables
load_dotenv()

# 2. CUSTOM CSS THEME
CUSTOM_CSS = """
<style>
/* Base typography */
h1 { font-weight: 700; }
h2, h3, h4 { font-weight: 600; }

/* Sidebar styling */
/* Removed hardcoded background so Streamlit native dark/light themes work */

/* Metric boxes */
[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    border: 1px solid var(--secondary-background-color);
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Pill tags for skills */
.skill-pill {
    display: inline-block;
    padding: 4px 12px;
    margin: 4px 4px 4px 0;
    border-radius: 16px;
    font-size: 0.9em;
    font-weight: 500;
}
.skill-match {
    background-color: #e6f4ea;
    color: #1e8e3e;
    border: 1px solid #ceead6;
}
.skill-missing {
    background-color: #fce8e6;
    color: #d93025;
    border: 1px solid #fad2cf;
}

/* Score display */
.score-display {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0;
    padding-bottom: 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper function for rendering skill pills
def render_skills(skills_list, match_type):
    if not skills_list:
        st.write("None")
        return
        
    pills_html = ""
    css_class = "skill-match" if match_type == "match" else "skill-missing"
    for skill in skills_list:
        pills_html += f'<span class="skill-pill {css_class}">{skill}</span>'
    st.markdown(pills_html, unsafe_allow_html=True)

def get_score_color(score):
    if score >= 80:
        return "#1e8e3e"  # Green
    elif score >= 60:
        return "#f9ab00"  # Yellow
    else:
        return "#d93025"  # Red

def highlight_keywords(text, keywords):
    if not text:
        return ""
    escaped_text = html.escape(text)
    
    if not keywords:
        return escaped_text.replace('\n', '<br>')
        
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    for kw in sorted_keywords:
        if not kw.strip(): continue
        pattern = re.compile(r'\b(' + re.escape(html.escape(kw)) + r')\b', re.IGNORECASE)
        escaped_text = pattern.sub(r"<span style='background-color: rgba(74, 222, 128, 0.25); padding: 2px 4px; border-radius: 4px; font-weight: 500;'>\1</span>", escaped_text)
        
    return escaped_text.replace('\n', '<br>')

st.title("📄 AI Resume Screening System")
st.markdown("Automatically match resumes against job descriptions and generate AI-driven feedback.")
st.divider()

# 7. SIDEBAR
with st.sidebar:
    st.markdown("## 🎛️ Dashboard Controls")
    
    st.subheader("Scoring Weights")
    tfidf_weight = st.slider("Keyword Weight (TF-IDF)", 0.0, 1.0, 0.33, 0.1, help="Measures exact keyword overlaps between the job description and the resume.")
    semantic_weight = st.slider("Semantic Weight (AI)", 0.0, 1.0, 0.33, 0.1, help="Uses AI to understand the meaning of the text, catching synonyms (e.g., 'ML' and 'Machine Learning').")
    skill_weight = st.slider("Skill Overlap Weight", 0.0, 1.0, 0.34, 0.1, help="Matches specifically extracted technical and soft skills based on an industry database.")
    
    # Normalize weights
    total_weight = tfidf_weight + semantic_weight + skill_weight
    if total_weight > 0:
        tfidf_weight /= total_weight
        semantic_weight /= total_weight
        skill_weight /= total_weight
        
    st.divider()
    st.subheader("Fairness Options")
    enable_bias_mitigation = st.checkbox("Score using bias-reduced text", help="Removes names, gendered pronouns, and university names before scoring to reduce unconscious bias.")

# 5. LAYOUT & SPACING
st.info("ℹ️ **Fairness & Bias:** This tool scores resumes based only on skills and job-relevant text. It does not use name, gender, age, or university prestige as scoring factors. Toggle 'bias-reduced scoring' in the sidebar to compare results with identifying details removed.")

with st.container(border=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Job Description")
        job_desc = st.text_area("Paste the job description here:", height=200, label_visibility="collapsed")
    
    with col2:
        st.subheader("2. Resumes")
        uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed")

if st.button("🚀 Analyze Resumes", type="primary", use_container_width=True):
    if not job_desc:
        st.error("Please provide a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Processing text and extracting skills..."):
            processed_jd = clean_text(job_desc)
            
            resumes_data = []
            results = []
            
            for file in uploaded_files:
                try:
                    raw_text = extract_text(file, file.name)
                    processed_text = clean_text(raw_text)
                    
                    score_data = calculate_ats_score(processed_text, processed_jd, tfidf_weight, semantic_weight, skill_weight)
                    
                    bias_score = None
                    if enable_bias_mitigation:
                        sanitized_raw = sanitize_for_bias(raw_text)
                        processed_sanitized = clean_text(sanitized_raw)
                        bias_score_data = calculate_ats_score(processed_sanitized, processed_jd, tfidf_weight, semantic_weight, skill_weight)
                        bias_score = bias_score_data["final_score"]
                    
                    resumes_data.append({
                        "filename": file.name,
                        "raw_text": raw_text,
                        "processed_text": processed_text
                    })
                    
                    result_entry = {
                        "Candidate / File": file.name,
                        "TF-IDF Score": score_data["tfidf_score"],
                        "Semantic Score": score_data["semantic_score"],
                        "Skill Score": score_data["skill_score"],
                        "Final ATS Score": score_data["final_score"],
                        "Matched Skills List": score_data["matched_skills"],
                        "Missing Skills List": score_data["missing_skills"],
                        "TF-IDF Matched Words": score_data.get("tfidf_matched_words", []),
                        "Semantic Only Matches": score_data.get("semantic_only_matches", []),
                        "Explainability Summary": score_data.get("explainability_summary", "")
                    }
                    
                    if enable_bias_mitigation:
                        result_entry["Bias-Reduced Score"] = bias_score
                        
                    results.append(result_entry)
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
            
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(by='Final ATS Score', ascending=False).reset_index(drop=True)
                
                st.session_state['results_df'] = results_df
                st.session_state['resumes_data'] = resumes_data
                st.session_state['job_desc'] = job_desc
                st.success("Analysis complete!")

# Display Results
if 'results_df' in st.session_state:
    st.divider()
    
    with st.container(border=True):
        st.subheader("🏆 Candidate Rankings Overview")
        
        # 6. RANKING TABLE - Red to Green gradient
        cols_to_show = ['Candidate / File', 'TF-IDF Score', 'Semantic Score', 'Skill Score', 'Final ATS Score']
        format_dict = {
            "TF-IDF Score": "{:.2f}%",
            "Semantic Score": "{:.2f}%",
            "Skill Score": "{:.2f}%",
            "Final ATS Score": "{:.2f}%"
        }
        
        if enable_bias_mitigation:
            cols_to_show.append('Bias-Reduced Score')
            format_dict['Bias-Reduced Score'] = "{:.2f}%"
            
        overview_df = st.session_state['results_df'][cols_to_show]
        st.dataframe(
            overview_df.style.format(format_dict).background_gradient(subset=["Final ATS Score"], cmap="RdYlGn", vmin=0, vmax=100),
            use_container_width=True
        )
    
    st.divider()
    st.subheader("📋 Detailed Candidate Reports")
    
    for i, row in st.session_state['results_df'].iterrows():
        candidate_name = row['Candidate / File']
        final_score = row['Final ATS Score']
        matched_list = row['Matched Skills List']
        missing_list = row['Missing Skills List']
        
        color = get_score_color(final_score)
            
        with st.expander(f"{candidate_name} — Score: {final_score}%", expanded=(i==0)):
            # 3. SCORE DISPLAY
            if 'Bias-Reduced Score' in row and pd.notna(row['Bias-Reduced Score']):
                bias_score = row['Bias-Reduced Score']
                bias_color = get_score_color(bias_score)
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown("**Original Score**")
                    st.markdown(f"<div class='score-display' style='color:{color};'>{final_score}%</div>", unsafe_allow_html=True)
                    st.progress(min(int(final_score), 100))
                with sc2:
                    st.markdown("**Bias-Reduced Score**")
                    st.markdown(f"<div class='score-display' style='color:{bias_color};'>{bias_score}%</div>", unsafe_allow_html=True)
                    st.progress(min(int(bias_score), 100))
            else:
                st.markdown(f"<div class='score-display' style='color:{color};'>{final_score}%</div>", unsafe_allow_html=True)
                st.progress(min(int(final_score), 100))
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Score Breakdown Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("TF-IDF Score", f"{row['TF-IDF Score']}%")
            m2.metric("Semantic Score", f"{row['Semantic Score']}%")
            m3.metric("Skill Score", f"{row['Skill Score']}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Explainability Panel
            with st.container(border=True):
                st.markdown("#### 🔍 Why this score?")
                st.markdown(f"*{row['Explainability Summary']}*")
                
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    st.markdown("**Top TF-IDF Keywords:**")
                    if row['TF-IDF Matched Words']:
                        st.write(", ".join(row['TF-IDF Matched Words']))
                    else:
                        st.write("None")
                with ec2:
                    st.markdown("**Semantic Concepts:**")
                    if row['Semantic Only Matches']:
                        for match in row['Semantic Only Matches']:
                            st.write(f"- {match}")
                    else:
                        st.write("None")
                with ec3:
                    st.markdown("**Skill Overlap:**")
                    total_req = len(row['Matched Skills List']) + len(row['Missing Skills List'])
                    st.write(f"{len(row['Matched Skills List'])} out of {total_req} required skills found")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 4 & 5. SKILL TAGS in Columns
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### ✅ Matched Skills")
                render_skills(matched_list, "match")
            with c2:
                st.markdown("#### ❌ Missing Skills")
                render_skills(missing_list, "missing")
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.expander("📄 View Resume with Matched Keywords Highlighted"):
                candidate_text = next(r['raw_text'] for r in st.session_state['resumes_data'] if r['filename'] == candidate_name)
                highlighted_html = highlight_keywords(candidate_text, matched_list)
                st.markdown(highlighted_html, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### 🤖 AI Candidate Feedback")
            if st.button(f"Generate Insights for {candidate_name}", key=f"btn_ai_{i}", type="secondary"):
                with st.spinner("Analyzing candidate strengths and gaps (this takes a few seconds)..."):
                    candidate_text = next(r['raw_text'] for r in st.session_state['resumes_data'] if r['filename'] == candidate_name)
                    key_to_use = os.getenv("GEMINI_API_KEY")
                    
                    feedback = generate_feedback(candidate_text, st.session_state['job_desc'], api_key=key_to_use)
                    
                    with st.container(border=True):
                        st.markdown(feedback)
                    
    st.divider()
