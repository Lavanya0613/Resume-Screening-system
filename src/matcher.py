import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model once to save time
# 'all-MiniLM-L6-v2' is lightweight and fast
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_tfidf_similarity(job_desc: str, resumes: list[str]) -> list[float]:
    """
    Computes cosine similarity using TF-IDF.
    """
    if not resumes:
        return []
    
    vectorizer = TfidfVectorizer()
    # Fit and transform the job description and all resumes
    vectors = vectorizer.fit_transform([job_desc] + resumes)
    
    # The first vector is the job description
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    
    # Compute similarity between job desc and each resume
    similarities = cosine_similarity(job_vector, resume_vectors).flatten()
    return similarities.tolist()

def compute_semantic_similarity(job_desc: str, resumes: list[str]) -> list[float]:
    """
    Computes cosine similarity using Sentence Transformers.
    Better for catching synonyms and related concepts.
    """
    if not resumes:
        return []
    
    # Encode sentences to get embeddings
    job_embedding = semantic_model.encode(job_desc, convert_to_tensor=True)
    resume_embeddings = semantic_model.encode(resumes, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.cos_sim(job_embedding, resume_embeddings)[0]
    return cosine_scores.tolist()

def rank_resumes(job_desc_processed: str, resumes_data: list[dict], weight_tfidf: float = 0.5, weight_semantic: float = 0.5) -> pd.DataFrame:
    """
    Takes the processed job description and a list of resume dictionaries.
    resume dictionary should have:
    - 'filename': str
    - 'processed_text': str
    
    Returns a pandas DataFrame sorted by combined similarity score.
    """
    if not resumes_data:
        return pd.DataFrame()
        
    resume_texts = [r['processed_text'] for r in resumes_data]
    
    tfidf_scores = compute_tfidf_similarity(job_desc_processed, resume_texts)
    semantic_scores = compute_semantic_similarity(job_desc_processed, resume_texts)
    
    results = []
    for i, resume in enumerate(resumes_data):
        combined_score = (tfidf_scores[i] * weight_tfidf) + (semantic_scores[i] * weight_semantic)
        results.append({
            'Candidate / File': resume['filename'],
            'TF-IDF Score': round(tfidf_scores[i] * 100, 2),
            'Semantic Score': round(semantic_scores[i] * 100, 2),
            'Match Percentage': round(combined_score * 100, 2)
        })
        
    df = pd.DataFrame(results)
    # Sort by Match Percentage descending
    df = df.sort_values(by='Match Percentage', ascending=False).reset_index(drop=True)
    
    return df
