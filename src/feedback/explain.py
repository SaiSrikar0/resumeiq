import os
import json
import pickle
import torch
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.classical_baseline import check_degree_match
from src.models.embedding_model import get_attention_explainability, load_model

def load_baseline():
    """Loads the best-performing classical ML model and its feature importances."""
    metadata_path = 'src/models/artifacts/baseline_metadata.json'
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Baseline metadata not found. Run classical_baseline.py first.")
        
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
        
    best_type = meta['best_model_type']
    importances = meta['feature_importances']
    
    if best_type == 'xgboost':
        import xgboost as xgb
        clf = xgb.XGBClassifier()
        clf.load_model('src/models/artifacts/baseline_xgb.json')
    else:
        import lightgbm as lgb
        clf = lgb.Booster(model_file='src/models/artifacts/baseline_lgb.txt')
        
    return clf, best_type, importances

def get_fit_score(clf, model_type, features_dict) -> float:
    """Computes baseline model prediction (probability of match scaled to 0-100)."""
    # Features must match the training order:
    # ['skill_overlap_ratio', 'experience_gap', 'degree_match', 'tfidf_similarity']
    features_list = [
        features_dict['skill_overlap_ratio'],
        features_dict['experience_gap'],
        features_dict['degree_match'],
        features_dict['tfidf_similarity']
    ]
    df = pd.DataFrame([features_list], columns=[
        'skill_overlap_ratio', 'experience_gap', 'degree_match', 'tfidf_similarity'
    ])
    
    if model_type == 'xgboost':
        prob = float(clf.predict_proba(df)[:, 1][0])
    else:
        prob = float(clf.predict(df)[0])
        
    return prob * 100.0

def generate_explanation(explain_obj) -> str:
    """
    Generates a 3-5 sentence plain-language explanation of why a candidate fits or does not fit a JD,
    grounded only in the actual computed gaps.
    """
    fit_score = explain_obj['baseline_fit_score']
    overlap = explain_obj['skill_overlap_ratio']
    matched_skills = explain_obj['matched_skills']
    missing_skills = explain_obj['missing_skills']
    exp_gap = explain_obj['experience_gap']
    cand_exp = explain_obj['candidate_experience']
    req_exp = explain_obj['required_experience']
    edu_match = explain_obj['education_fit']['is_match']
    cand_edu = explain_obj['education_fit']['candidate_education']
    req_edu = explain_obj['education_fit']['required_education']
    top_tokens = explain_obj['attention_top_tokens']
    
    sentences = []
    
    # 1. Overall fit score sentence
    sentences.append(
        f"ResumeIQ calculated a fit score of {fit_score:.1f}% for this role, driven primarily by "
        f"a skill overlap of {overlap * 100:.0f}% and semantic text similarity."
    )
    
    # 2. Skill overlap sentence
    if matched_skills:
        match_str = ", ".join(matched_skills[:3])
        if len(matched_skills) > 3:
            match_str += f", and {len(matched_skills) - 3} other(s)"
        skill_sentence = f"The candidate matches core skills such as {match_str}."
    else:
        skill_sentence = "The candidate's resume does not overlap with any explicitly required skills."
        
    if missing_skills:
        miss_str = ", ".join(missing_skills[:3])
        if len(missing_skills) > 3:
            miss_str += f", and {len(missing_skills) - 3} other(s)"
        skill_sentence += f" However, key required skills like {miss_str} are missing."
    else:
        skill_sentence += " They possess all required technical skills listed."
    sentences.append(skill_sentence)
    
    # 3. Experience sentence
    if exp_gap > 0:
        sentences.append(
            f"There is an experience gap of {exp_gap:.1f} years: the candidate has {cand_exp:.1f} years, "
            f"while the role requires {req_exp:.1f} years."
        )
    else:
        sentences.append(
            f"The candidate's {cand_exp:.1f} years of experience successfully meets the required {req_exp:.1f} years."
        )
        
    # 4. Education & Attention sentence
    edu_status = "meets or exceeds" if edu_match else "does not meet"
    attention_str = ", ".join(top_tokens[:3])
    sentences.append(
        f"The candidate's education ({cand_edu}) {edu_status} the required {req_edu}. "
        f"Additionally, the neural model's attention highlights that terminology surrounding "
        f"'{attention_str}' in the resume was highly influential during matching."
    )
    
    return " ".join(sentences)

def compute_explainability_breakdown(resume_row, jd_row, baseline_clf, baseline_type, importances, tfidf_vectorizer, embedding_model, embedding_tokenizer):
    """
    Computes a structured explainability object combining ML features, baseline scores,
    and transformer attention weights.
    """
    # 1. Extract raw skills
    cand_skills = resume_row.get('ExtractedSkills', [])
    req_skills = jd_row.get('RequiredSkills', [])
    
    cand_skills_set = set(s.lower() for s in cand_skills)
    req_skills_set = set(s.lower() for s in req_skills)
    
    matched_skills = sorted(list(set(s for s in req_skills if s.lower() in cand_skills_set)))
    missing_skills = sorted(list(set(s for s in req_skills if s.lower() not in cand_skills_set)))
    
    skill_overlap = len(cand_skills_set & req_skills_set) / len(req_skills_set) if req_skills_set else 0.0
    
    # 2. Experience gap
    cand_exp = float(resume_row.get('YearsOfExperience', 0.0))
    req_exp = float(jd_row.get('RequiredExperience', 0.0))
    exp_gap = req_exp - cand_exp
    
    # 3. Education fit
    cand_edu = resume_row.get('DegreeLevel', 'None')
    req_edu = jd_row.get('RequiredEducation', 'None')
    edu_match = check_degree_match(cand_edu, req_edu) == 1.0
    
    # 4. TF-IDF Similarity
    res_text = resume_row.get('Text', '')
    jd_text = jd_row.get('Description', '')
    
    from sklearn.preprocessing import normalize
    res_vec = tfidf_vectorizer.transform([res_text])
    jd_vec = tfidf_vectorizer.transform([jd_text])
    res_norm = normalize(res_vec, norm='l2', axis=1)
    jd_norm = normalize(jd_vec, norm='l2', axis=1)
    tfidf_sim = float((res_norm * jd_norm.T)[0, 0])
    
    # 5. Predict Fit Score
    feats = {
        'skill_overlap_ratio': skill_overlap,
        'experience_gap': exp_gap,
        'degree_match': 1.0 if edu_match else 0.0,
        'tfidf_similarity': tfidf_sim
    }
    model_score = get_fit_score(baseline_clf, baseline_type, feats)
    
    # Rebalance score: 40% skills, 30% experience, 15% degree match, 15% model semantic score
    skill_score = skill_overlap
    if exp_gap <= 0:
        exp_score = 1.0
    else:
        exp_score = max(0.0, 1.0 - (exp_gap / req_exp)) if req_exp > 0 else 0.0
        
    deg_score = 1.0 if edu_match else 0.0
    semantic_score = model_score / 100.0
    
    fit_score = 0.40 * skill_score + 0.30 * exp_score + 0.15 * deg_score + 0.15 * semantic_score
    fit_score = float(fit_score * 100.0)
    
    # 6. Attention Weights Explainability
    # Embed JD first
    from src.models.embedding_model import get_embeddings_batched
    best_pooling = 'mean' # From embedding metadata
    jd_emb = get_embeddings_batched([jd_text], embedding_model, embedding_tokenizer, pooling=best_pooling, batch_size=1)[0]
    
    attn_explain = get_attention_explainability(res_text, jd_emb, embedding_model, embedding_tokenizer)
    
    # Combine everything
    explain_obj = {
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'skill_overlap_ratio': skill_overlap,
        'experience_gap': exp_gap,
        'candidate_experience': cand_exp,
        'required_experience': req_exp,
        'education_fit': {
            'candidate_education': cand_edu,
            'required_education': req_edu,
            'is_match': edu_match
        },
        'baseline_fit_score': fit_score,
        'baseline_feature_importances': importances,
        'attention_top_tokens': attn_explain['top_tokens'],
        'attention_note': attn_explain['explainability_note']
    }
    
    # Generate natural language explanation
    explain_obj['explanation_text'] = generate_explanation(explain_obj)
    
    return explain_obj

def main():
    print("Testing Explainability Breakdown...")
    # Load all models
    baseline_clf, baseline_type, importances = load_baseline()
    
    with open('src/models/artifacts/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        
    tokenizer, model = load_model()
    
    # Load sample resume & JD
    resumes_df = pd.read_parquet('data/processed_resumes.parquet')
    jds_df = pd.read_json('data/job_descriptions.jsonl', lines=True)
    
    res = resumes_df.iloc[0]
    jd = jds_df.iloc[0]
    
    breakdown = compute_explainability_breakdown(res, jd, baseline_clf, baseline_type, importances, tfidf, model, tokenizer)
    print("\nStructured Explanation Object:")
    print(json.dumps(breakdown, indent=4))
    
if __name__ == '__main__':
    main()
