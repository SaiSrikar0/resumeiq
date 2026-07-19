import os
import json
import pickle
import numpy as np
import pandas as pd
import pytest

from src.models import check_degree_match, build_features_optimized, load_model
from src.feedback import compute_explainability_breakdown, load_baseline, generate_feedback_report
from src.retrieval import load_index

def test_check_degree_match():
    # Priority: PhD (3) > Master's (2) > Bachelor's (1) > None (0)
    assert check_degree_match("PhD", "Master's") == 1.0
    assert check_degree_match("Master's", "Master's") == 1.0
    assert check_degree_match("Bachelor's", "Master's") == 0.0
    assert check_degree_match("None", "Bachelor's") == 0.0

def test_build_features_optimized():
    res_row = {'ExtractedSkills': ['Python', 'SQL', 'Git'], 'YearsOfExperience': 3.5, 'DegreeLevel': "Master's"}
    jd_row = {'RequiredSkills': ['Python', 'Docker'], 'RequiredExperience': 5.0, 'RequiredEducation': "Bachelor's"}
    
    # Precomputed sparse matrices mock (1 feature element)
    # Norm vectors of shape [1, 5000]
    from scipy.sparse import csr_matrix
    res_vec = csr_matrix(([1.0], ([0], [0])), shape=(1, 5000))
    jd_vec = csr_matrix(([1.0], ([0], [0])), shape=(1, 5000))
    
    feats = build_features_optimized(res_row, jd_row, 0, 0, res_vec, jd_vec)
    
    # Intersection of skills: {'python'} (Docker is missing). Required: {'python', 'docker'}
    # Overlap ratio: 1/2 = 0.5
    assert feats['skill_overlap_ratio'] == 0.5
    # Experience gap: 5.0 - 3.5 = 1.5
    assert feats['experience_gap'] == 1.5
    # Degree match: Master's >= Bachelor's -> 1.0
    assert feats['degree_match'] == 1.0
    # Cosine Similarity (dot product): 1.0
    assert feats['tfidf_similarity'] == 1.0

def test_explain_and_generate_report():
    # Load sample inputs
    resumes_df = pd.read_parquet('data/processed/processed_resumes.parquet')
    jds_df = pd.read_json('data/processed/job_descriptions.jsonl', lines=True)
    
    res = resumes_df.iloc[0]
    jd = jds_df.iloc[0]
    
    # Load model and vectorizer
    baseline_clf, baseline_type, importances = load_baseline()
    with open('artifacts/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        
    tokenizer, model = load_model()
    faiss_db = load_index(model, tokenizer)
    
    # 1. Test Explainability Breakdown
    breakdown = compute_explainability_breakdown(
        res, jd, baseline_clf, baseline_type, importances, tfidf, model, tokenizer
    )
    
    assert 'baseline_fit_score' in breakdown
    assert 0 <= breakdown['baseline_fit_score'] <= 100
    assert 'explanation_text' in breakdown
    assert len(breakdown['explanation_text'].split('.')) >= 3 # at least 3 sentences
    assert 'attention_top_tokens' in breakdown
    assert len(breakdown['attention_top_tokens']) > 0
    
    # 2. Test Feedback Generation
    feedback = generate_feedback_report(
        res, jd, breakdown, faiss_db, resumes_df, baseline_clf, baseline_type, tfidf, use_llm=False
    )
    
    assert "# ResumeIQ Feedback Report" in feedback
    assert "## Score & Explainability Summary" in feedback
    assert "[Addresses Gap:" in feedback # Must cite gaps
    assert "## Peer References" in feedback
