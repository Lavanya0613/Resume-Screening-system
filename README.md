# AI Resume Screening System 📄

An intelligent, Streamlit-based application designed to help recruiters and hiring managers automatically screen and rank candidate resumes against a job description. 

It leverages traditional NLP (TF-IDF), modern Semantic AI (Sentence Transformers), and Generative AI (Anthropic Claude) to provide a comprehensive analysis of candidate fit.

---

## 🌟 Key Features

1. **Multi-Resume Uploading**: Easily parse multiple PDF and DOCX files at once.
2. **Robust Text Extraction**: Handles text extraction automatically, and gracefully flags scanned/image-based PDFs or corrupted files.
3. **Advanced Tri-Scoring Engine**:
   - **TF-IDF**: Base keyword frequency matching.
   - **Semantic Similarity**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`) catch synonymous concepts (e.g. "ML" vs "Machine Learning").
   - **Skill Overlap**: Automatically extracts and compares required skills against a database of 150+ industry-standard technical and soft skills.
4. **Smart Skill Extraction**: Uses **spaCy** for Named Entity Recognition (NER) and lemmatization to dynamically catch missing technologies.
5. **Generative AI Feedback**: Deep integration with the **Anthropic Claude API** to generate actionable feedback on strengths, gaps, and interview suggestions for any candidate.

---

## 🛠️ Tech Stack

- **Frontend Interface**: Streamlit
- **Text Parsing**: `pdfplumber`, `python-docx`
- **Text Processing & NLP**: `nltk`, `spacy`
- **Vectorization & Scoring**: `scikit-learn`, `sentence-transformers`
- **Generative AI**: `anthropic`
- **Data Handling**: `pandas`

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Lavanya0613/Resume-Screening-system.git
cd Resume-Screening-system
```

### 2. Set up the Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configure API Keys
1. Rename `.env.example` to `.env`.
2. Add your Anthropic API Key inside `.env`:
```text
ANTHROPIC_API_KEY=your_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to use the system!

---

## 📸 Screenshots

*(Add a screenshot of your ranked candidate dashboard here!)*
![App Screenshot Placeholder](https://via.placeholder.com/800x400.png?text=AI+Resume+Screening+Dashboard)

## Fairness & Bias Considerations

In automated ATS systems, algorithms can inadvertently discriminate based on proxies for race, gender, or socioeconomic status (like names or university prestige). 

To mitigate this, this project includes an optional **Bias-Reduced Scoring** mode.
When toggled:
- **Names** are stripped using Named Entity Recognition (spaCy).
- **Gendered Pronouns** (he/she/him/her) are removed.
- **Institution Names** (University/College) are generalized.

**Limitations:** This is a basic proof-of-concept mitigation, not a comprehensive fairness audit. Machine learning models (especially semantic embeddings) can still infer demographics through subtle language proxies. Always pair AI screening with human review.
