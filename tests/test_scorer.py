import pytest
from unittest.mock import MagicMock

# We must mock Streamlit entirely so it doesn't try to run in a test context
@pytest.fixture(autouse=True)
def mock_streamlit(mocker):
    # Mock st.cache_resource and st.cache_data to just return the function they wrap
    def mock_cache_decorator(func):
        return func
        
    mocker.patch('streamlit.cache_resource', side_effect=lambda x=None: mock_cache_decorator if x is None else mock_cache_decorator(x))
    mocker.patch('streamlit.cache_data', side_effect=lambda x=None: mock_cache_decorator if x is None else mock_cache_decorator(x))
    mocker.patch('streamlit.spinner', return_value=MagicMock())

@pytest.fixture(autouse=True)
def mock_semantic_model(mocker):
    # Mock the semantic model to prevent loading sentence-transformers during tests
    mock_model = MagicMock()
    # When encode is called, return a dummy list (util.cos_sim can handle lists/arrays)
    mock_model.encode.return_value = [[1.0, 0.0]] 
    mocker.patch('src.scorer.get_semantic_model', return_value=mock_model)
    
    import torch
    mocker.patch('src.scorer.util.cos_sim', return_value=torch.tensor([[1.0]]))
    
def test_scorer_identical_texts():
    from src.scorer import calculate_ats_score
    
    text = "Software Engineer with Python and SQL experience."
    result = calculate_ats_score(text, text, weight_tfidf=0.33, weight_semantic=0.33, weight_skills=0.34)
    
    # Since texts are identical, tfidf should be 100
    assert result["tfidf_score"] == 100.0
    # Semantic score will be 100 because our mock cos_sim returns 1.0
    assert result["semantic_score"] == 100.0
    # Final score should be > 0
    assert result["final_score"] > 0.0
    
def test_scorer_unrelated_texts(mocker):
    from src.scorer import calculate_ats_score
    
    # Override the default mock to return 0.0 for unrelated
    import torch
    mocker.patch('src.scorer.util.cos_sim', return_value=torch.tensor([[0.0]]))
    
    text1 = "Python developer data science"
    text2 = "Chef cooking culinary arts"
    
    result = calculate_ats_score(text1, text2, weight_tfidf=0.33, weight_semantic=0.33, weight_skills=0.34)
    
    # TF-IDF should be 0 because no overlapping words
    assert result["tfidf_score"] == 0.0
    # Semantic should be 0 because of our override
    assert result["semantic_score"] == 0.0
    
def test_scorer_blending(mocker):
    from src.scorer import calculate_ats_score
    
    # Force semantic score to be 50, TF-IDF to 0
    import torch
    mocker.patch('src.scorer.util.cos_sim', return_value=torch.tensor([[0.5]]))
    
    text1 = "Software Engineer"
    text2 = "Data Scientist"
    
    # Weights: TFIDF 0%, Semantic 100%, Skills 0%
    result = calculate_ats_score(text1, text2, weight_tfidf=0.0, weight_semantic=1.0, weight_skills=0.0)
    
    assert result["semantic_score"] == 50.0
    # Final score strictly matches semantic due to 1.0 weight
    assert result["final_score"] == 50.0
