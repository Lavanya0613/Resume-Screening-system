import os
import anthropic
from dotenv import load_dotenv

# Load env variables if they exist
load_dotenv()

def generate_feedback(resume_text: str, job_desc: str, api_key: str = None) -> str:
    """
    Calls the Anthropic API to generate feedback on a candidate's resume
    against a job description.
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    
    if not key:
        return "Anthropic API Key not provided. Please add it to .env or pass it directly."
        
    client = anthropic.Anthropic(api_key=key)
    
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

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error generating feedback: {str(e)}"
