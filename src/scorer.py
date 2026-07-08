from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.skill_matcher import match_skills
from sentence_transformers import SentenceTransformer, util

# Load the Semantic model once
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_ats_score(cleaned_resume: str, cleaned_jd: str, weight_tfidf: float = 0.33, weight_semantic: float = 0.33, weight_skills: float = 0.34) -> dict:
    """
    Calculates a blended ATS score using TF-IDF, Semantic Similarity (Sentence Transformers),
    and skill overlap percentage.
    """
    if not cleaned_resume or not cleaned_jd:
        return {
            "final_score": 0.0,
            "tfidf_score": 0.0,
            "semantic_score": 0.0,
            "skill_score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "jd_skills": [],
            "resume_skills": [],
            "tfidf_matched_words": [],
            "semantic_only_matches": [],
            "explainability_summary": "No text provided to analyze."
        }
        
    # 1. TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_jd, cleaned_resume])
    cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
    tfidf_score = round(cosine_sim * 100, 2)
    
    # Extract TF-IDF matched keywords
    feature_names = vectorizer.get_feature_names_out()
    jd_vec = vectors[0].toarray()[0]
    res_vec = vectors[1].toarray()[0]
    matched_indices = [i for i in range(len(feature_names)) if jd_vec[i] > 0 and res_vec[i] > 0]
    matched_words = sorted([(feature_names[i], res_vec[i]) for i in matched_indices], key=lambda x: x[1], reverse=True)
    top_tfidf_words = [w[0] for w in matched_words[:10]]
        
    # 2. Semantic Similarity
    job_emb = semantic_model.encode(cleaned_jd, convert_to_tensor=True)
    res_emb = semantic_model.encode(cleaned_resume, convert_to_tensor=True)
    semantic_sim = util.cos_sim(job_emb, res_emb)[0].item()
    semantic_score = round(semantic_sim * 100, 2)

    # 3. Skill Overlap
    skill_metrics = match_skills(cleaned_resume, cleaned_jd)
    skill_score = skill_metrics['overlap_percentage']
    
    # 4. Blended Score
    final_score = round(
        (tfidf_score * weight_tfidf) + 
        (semantic_score * weight_semantic) + 
        (skill_score * weight_skills), 
    2)
    
    # 5. Semantic-Only Matches (Explainability)
    missing_jd_skills = skill_metrics["missing_skills"]
    unmatched_resume_skills = [s for s in skill_metrics["resume_skills"] if s not in skill_metrics["matched_skills"]]
    
    semantic_only_matches = []
    if missing_jd_skills and unmatched_resume_skills:
        jd_skill_embs = semantic_model.encode(missing_jd_skills, convert_to_tensor=True)
        res_skill_embs = semantic_model.encode(unmatched_resume_skills, convert_to_tensor=True)
        sim_matrix = util.cos_sim(jd_skill_embs, res_skill_embs)
        
        for i, jd_s in enumerate(missing_jd_skills):
            for j, res_s in enumerate(unmatched_resume_skills):
                if sim_matrix[i][j] > 0.65:
                    semantic_only_matches.append(f"{res_s} ≈ {jd_s}")
    
    semantic_only_matches = list(set(semantic_only_matches))
    
    # 6. Summary
    total_req = len(skill_metrics['jd_skills'])
    found_req = len(skill_metrics['matched_skills'])
    summary = f"This resume matched {len(matched_words)} keywords directly, with {len(semantic_only_matches)} conceptually related skills detected. It meets {found_req} out of {total_req} required skills."

    return {
        "final_score": final_score,
        "tfidf_score": tfidf_score,
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "matched_skills": skill_metrics["matched_skills"],
        "missing_skills": skill_metrics["missing_skills"],
        "jd_skills": skill_metrics["jd_skills"],
        "resume_skills": skill_metrics["resume_skills"],
        "tfidf_matched_words": top_tfidf_words,
        "semantic_only_matches": semantic_only_matches,
        "explainability_summary": summary
    }
