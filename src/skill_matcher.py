import json
import os
import re
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: en_core_web_sm not found. Falling back to basic matching.")
    nlp = None

# Load skills dataset once
SKILLS_FILE = os.path.join(os.path.dirname(__file__), 'skills.json')
with open(SKILLS_FILE, 'r') as f:
    SKILLS_DATA = json.load(f)

# Flatten skills for easier matching, store in a set (lowercase for case-insensitive matching)
ALL_SKILLS = set()
for category, skills in SKILLS_DATA.items():
    for skill in skills:
        ALL_SKILLS.add(skill.lower())

def extract_skills(cleaned_text: str) -> set:
    """
    Extracts known skills from the given cleaned text.
    Uses regex for predefined skills and spaCy for lemmatization/NER.
    """
    found_skills = set()
    
    # Optional spaCy processing for lemmatization and NER
    text_to_search = cleaned_text
    if nlp:
        doc = nlp(cleaned_text)
        # Reconstruct text using lemmas for better matching (e.g., managing -> manage)
        text_to_search = " ".join([token.lemma_ for token in doc])
        
        # Simple NER to catch potential technologies/tools not in predefined list
        for ent in doc.ents:
            # Tech tools are often misclassified as ORG or PRODUCT or PERSON by basic models
            if ent.label_ in ['ORG', 'PRODUCT']:
                # Filter out obvious noise (very short or very long entities)
                if 2 < len(ent.text) < 20:
                    found_skills.add(ent.text.lower())
    
    # Regex matching against predefined skills database
    for skill in ALL_SKILLS:
        # Create a regex pattern with word boundaries. 
        # Escape the skill in case it contains regex special chars (like C++)
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_to_search, flags=re.IGNORECASE):
            found_skills.add(skill)
            
    return found_skills

def match_skills(cleaned_resume: str, cleaned_jd: str) -> dict:
    """
    Matches skills from resume and job description.
    Returns matched skills (present in both) and missing skills (in JD but not resume).
    """
    resume_skills = extract_skills(cleaned_resume)
    jd_skills = extract_skills(cleaned_jd)
    
    matched_skills = jd_skills.intersection(resume_skills)
    missing_skills = jd_skills.difference(resume_skills)
    
    # Calculate overlap percentage
    if not jd_skills:
        overlap_percent = 100.0  # If no skills required, it's a perfect match
    else:
        overlap_percent = (len(matched_skills) / len(jd_skills)) * 100.0
        
    # Title case the skills for better UI presentation
    return {
        "jd_skills": [s.title() for s in jd_skills],
        "resume_skills": [s.title() for s in resume_skills],
        "matched_skills": [s.title() for s in matched_skills],
        "missing_skills": [s.title() for s in missing_skills],
        "overlap_percentage": round(overlap_percent, 2)
    }
