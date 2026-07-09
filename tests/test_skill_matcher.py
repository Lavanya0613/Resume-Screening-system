from src.skill_matcher import extract_skills, match_skills

def test_extract_skills_found():
    text = "I have experience with Python, SQL, and Machine Learning."
    skills = extract_skills(text)
    
    assert "python" in [s.lower() for s in skills]
    assert "sql" in [s.lower() for s in skills]

def test_extract_skills_case_insensitive():
    text = "Worked with PYTHON and sql."
    skills = extract_skills(text)
    
    assert "python" in [s.lower() for s in skills]
    assert "sql" in [s.lower() for s in skills]

def test_match_skills_missing():
    jd = "Seeking a developer with Python, Java, and AWS."
    resume = "I know Python and C++."
    
    result = match_skills(resume, jd)
    
    # JD wants Python, Java, AWS. Resume has Python.
    # Missing: Java, AWS. Matched: Python
    matched = [s.lower() for s in result["matched_skills"]]
    missing = [s.lower() for s in result["missing_skills"]]
    
    assert "python" in matched
    assert "java" in missing
    assert "aws" in missing
    assert "c++" not in matched
