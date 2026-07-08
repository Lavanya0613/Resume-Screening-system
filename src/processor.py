import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (runs once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def clean_text(text: str) -> str:
    """
    Cleans raw text by removing URLs, special characters, numbers, 
    extra whitespace, and common English stopwords.
    """
    if not text:
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove special characters and numbers (keep only alphabets)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(filtered_tokens)

# Load spaCy model for NER (used in bias mitigation)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    nlp = None

def sanitize_for_bias(text: str) -> str:
    """
    Strips potentially biasing information such as names, gendered pronouns, 
    and university/college names.
    """
    if not text:
        return ""
        
    sanitized = text
    
    # 1. Strip names using spaCy NER (if available)
    if nlp:
        doc = nlp(sanitized)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                sanitized = sanitized.replace(ent.text, "[NAME]")
                
    # 2. Strip gendered pronouns (case-insensitive regex)
    pronoun_pattern = r'\b(he|him|his|she|her|hers)\b'
    sanitized = re.sub(pronoun_pattern, '[PRONOUN]', sanitized, flags=re.IGNORECASE)
    
    # 3. Strip university/college names
    # Matches patterns like "University of X", "X University", "X College", "X Institute"
    institution_pattern = r'\b([A-Z][a-z]+ )?(University|College|Institute)( of [A-Z][a-z]+)?\b'
    sanitized = re.sub(institution_pattern, '[INSTITUTION]', sanitized)
    
    return sanitized
