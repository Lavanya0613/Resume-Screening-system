import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load env variables if they exist
load_dotenv()

def generate_feedback(resume_text: str, job_desc: str, api_key: str = None) -> str:
    """
    Calls the Google Gemini API to generate feedback on a candidate's resume
    against a job description.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    
    if not key:
        return "Gemini API Key not provided. Please add it to .env or pass it directly."
        
    try:
        client = genai.Client(api_key=key)
        
        prompt = f"""
        You are an expert technical recruiter and hiring manager. 
        Please review the following resume against the job description provided.
        Analyze the candidate's fit for the role.

        Job Description:
        {job_desc}

        Resume:
        {resume_text}

        Provide your feedback in the following format using Markdown:
        1. **Strengths:** (What makes the candidate a good fit)
        2. **Gaps:** (What is missing or weak compared to the JD)
        3. **Suggestions:** (How the candidate could improve their fit or what to ask in an interview)
        Keep it concise, professional, and directly related to the provided text.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error generating feedback: {str(e)}"
