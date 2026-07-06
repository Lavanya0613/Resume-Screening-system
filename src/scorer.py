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
            "resume_skills": []
        }
        
    # 1. TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_jd, cleaned_resume])
    cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
    tfidf_score = round(cosine_sim * 100, 2)
        
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
    
    return {
        "final_score": final_score,
        "tfidf_score": tfidf_score,
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "matched_skills": skill_metrics["matched_skills"],
        "missing_skills": skill_metrics["missing_skills"],
        "jd_skills": skill_metrics["jd_skills"],
        "resume_skills": skill_metrics["resume_skills"]
    }
