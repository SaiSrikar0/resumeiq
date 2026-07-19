import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# Degree Mapping
DEGREE_MAP = {
    'None': 0,
    "Bachelor's": 1,
    "Master's": 2,
    'PhD': 3
}

def check_degree_match(cand_deg: str, req_deg: str) -> float:
    cand_val = DEGREE_MAP.get(cand_deg, 0)
    req_val = DEGREE_MAP.get(req_deg, 0)
    return 1.0 if cand_val >= req_val else 0.0

def build_features_optimized(res_row, jd_row, res_idx, jd_idx, res_vecs_norm, jd_vecs_norm) -> dict:
    """Extract pairwise features for a single resume and JD pair using precomputed TF-IDF."""
    # 1. Skill Overlap Ratio
    cand_skills = set(s.lower() for s in res_row.get('ExtractedSkills', []))
    req_skills = set(s.lower() for s in jd_row.get('RequiredSkills', []))
    
    if req_skills:
        skill_overlap = len(cand_skills & req_skills) / len(req_skills)
    else:
        skill_overlap = 0.0
        
    # 2. Experience Gap
    cand_exp = float(res_row.get('YearsOfExperience', 0.0))
    req_exp = float(jd_row.get('RequiredExperience', 0.0))
    exp_gap = req_exp - cand_exp
    
    # 3. Degree Match
    degree_match = check_degree_match(res_row.get('DegreeLevel', 'None'), jd_row.get('RequiredEducation', 'None'))
    
    # 4. TF-IDF Cosine Similarity via dot product of normalized sparse vectors
    tfidf_sim = float((res_vecs_norm[res_idx] * jd_vecs_norm[jd_idx].T)[0, 0])
    
    return {
        'skill_overlap_ratio': skill_overlap,
        'experience_gap': exp_gap,
        'degree_match': degree_match,
        'tfidf_similarity': tfidf_sim
    }

def prepare_dataset(resumes_df, jds_df, res_vecs_norm, jd_vecs_norm, neg_ratio=3):
    """
    Constructs a dataset of pairs (resume, JD) with target labels (1 for same category, 0 for different).
    """
    records = []
    
    # Map IDs to indices for fast TF-IDF lookup
    res_id_to_idx = {res_id: i for i, res_id in enumerate(resumes_df['ResumeID'])}
    jd_id_to_idx = {jd_id: i for i, jd_id in enumerate(jds_df['JobDescriptionID'])}
    
    # Pre-build JD map by category
    jds_by_cat = {}
    all_jds = []
    for _, jd_row in jds_df.iterrows():
        cat = jd_row['Category']
        jds_by_cat.setdefault(cat, []).append(jd_row)
        all_jds.append(jd_row)
        
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Constructing positive and negative pairs...")
    for idx, res_row in resumes_df.iterrows():
        res_cat = res_row['Category']
        res_id = res_row['ResumeID']
        res_idx = res_id_to_idx[res_id]
        
        # 1. Positive pairs: JDs in the same category
        pos_jds = jds_by_cat.get(res_cat, [])
        for jd_row in pos_jds:
            jd_id = jd_row['JobDescriptionID']
            jd_idx = jd_id_to_idx[jd_id]
            feats = build_features_optimized(res_row, jd_row, res_idx, jd_idx, res_vecs_norm, jd_vecs_norm)
            feats.update({
                'ResumeID': res_id,
                'JobDescriptionID': jd_id,
                'label': 1.0
            })
            records.append(feats)
            
        # 2. Negative pairs: sample cross-category JDs
        cross_jds = [j for j in all_jds if j['Category'] != res_cat]
        if cross_jds:
            sampled_neg_jds = random.sample(cross_jds, min(len(cross_jds), neg_ratio))
            for jd_row in sampled_neg_jds:
                jd_id = jd_row['JobDescriptionID']
                jd_idx = jd_id_to_idx[jd_id]
                feats = build_features_optimized(res_row, jd_row, res_idx, jd_idx, res_vecs_norm, jd_vecs_norm)
                feats.update({
                    'ResumeID': res_id,
                    'JobDescriptionID': jd_id,
                    'label': 0.0
                })
                records.append(feats)
                
    return pd.DataFrame(records)

def run_cross_validation(df, features, model_type='xgboost'):
    """Performs 5-fold cross validation grouped by ResumeID to prevent leak."""
    unique_resumes = df['ResumeID'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    
    for train_idx, val_idx in kf.split(unique_resumes):
        train_ids = unique_resumes[train_idx]
        val_ids = unique_resumes[val_idx]
        
        train_df = df[df['ResumeID'].isin(train_ids)]
        val_df = df[df['ResumeID'].isin(val_ids)]
        
        X_train, y_train = train_df[features], train_df['label']
        X_val, y_val = val_df[features], val_df['label']
        
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)[:, 1]
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                verbosity=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)[:, 1]
        else:
            raise ValueError(f"Unknown model type {model_type}")
            
        precisions.append(precision_score(y_val, preds, zero_division=0))
        recalls.append(recall_score(y_val, preds, zero_division=0))
        f1s.append(f1_score(y_val, preds, zero_division=0))
        aucs.append(roc_auc_score(y_val, probs))
        
    return {
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
        'f1': float(np.mean(f1s)),
        'auc': float(np.mean(aucs))
    }

def main():
    # 1. Load data
    print("Loading datasets...")
    resumes_df = pd.read_parquet('data/processed/processed_resumes.parquet')
    jds_df = pd.read_json('data/processed/job_descriptions.jsonl', lines=True)
    
    # 2. Fit TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer...")
    corpus = pd.concat([
        resumes_df['Text'].fillna(''),
        jds_df['Description'].fillna('')
    ])
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_vectorizer.fit(corpus)
    
    # Pre-transform and L2-normalize
    res_vecs = tfidf_vectorizer.transform(resumes_df['Text'].fillna(''))
    jd_vecs = tfidf_vectorizer.transform(jds_df['Description'].fillna(''))
    
    res_vecs_norm = normalize(res_vecs, norm='l2', axis=1)
    jd_vecs_norm = normalize(jd_vecs, norm='l2', axis=1)
    
    # Save vectorizer
    os.makedirs('src/models/artifacts', exist_ok=True)
    with open('src/models/artifacts/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print("TF-IDF Vectorizer fit and saved to src/models/artifacts/tfidf_vectorizer.pkl")
    
    # 3. Construct paired dataset
    paired_df = prepare_dataset(resumes_df, jds_df, res_vecs_norm, jd_vecs_norm)
    print(f"Dataset constructed. Total pairs: {len(paired_df)}, Positive label ratio: {paired_df['label'].mean():.4f}")
    
    features = ['skill_overlap_ratio', 'experience_gap', 'degree_match', 'tfidf_similarity']
    
    # 4. Compare XGBoost and LightGBM
    print("Evaluating XGBoost...")
    xgb_metrics = run_cross_validation(paired_df, features, model_type='xgboost')
    print(f"XGBoost CV metrics: {xgb_metrics}")
    
    print("Evaluating LightGBM...")
    lgb_metrics = run_cross_validation(paired_df, features, model_type='lightgbm')
    print(f"LightGBM CV metrics: {lgb_metrics}")
    
    # Determine best model
    best_model_type = 'xgboost' if xgb_metrics['auc'] >= lgb_metrics['auc'] else 'lightgbm'
    print(f"\nBest model based on CV AUC: {best_model_type.upper()}")
    
    # 5. Train on full dataset
    X_full = paired_df[features]
    y_full = paired_df['label']
    
    if best_model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_full, y_full)
        
        # Save model
        model.save_model('src/models/artifacts/baseline_xgb.json')
        print("Trained XGBoost model saved to src/models/artifacts/baseline_xgb.json")
        
        # Get importances
        importance_scores = model.get_booster().get_score(importance_type='gain')
        # Map back to features checking both feature name and index
        feature_importances = {}
        for i, feat in enumerate(features):
            val = importance_scores.get(feat, importance_scores.get(f'f{i}', 0.0))
            feature_importances[feat] = float(val)
        # Normalize
        total_importance = sum(feature_importances.values())
        if total_importance > 0:
            feature_importances = {k: v / total_importance for k, v in feature_importances.items()}
            
    else:
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbosity=-1
        )
        model.fit(X_full, y_full)
        
        # Save model
        model.booster_.save_model('src/models/artifacts/baseline_lgb.txt')
        print("Trained LightGBM model saved to src/models/artifacts/baseline_lgb.txt")
        
        # Get importances
        importances = model.booster_.feature_importance(importance_type='gain')
        feature_importances = {feat: float(imp) for feat, imp in zip(features, importances)}
        total_importance = sum(feature_importances.values())
        if total_importance > 0:
            feature_importances = {k: v / total_importance for k, v in feature_importances.items()}
            
    # Save metadata including best model type, CV metrics and feature importances
    metadata = {
        'best_model_type': best_model_type,
        'xgb_cv_metrics': xgb_metrics,
        'lgb_cv_metrics': lgb_metrics,
        'feature_importances': feature_importances
    }
    
    with open('src/models/artifacts/baseline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Metadata (CV results & importances) saved to src/models/artifacts/baseline_metadata.json")
    print(f"Feature Importances (Gain-based): {feature_importances}")

if __name__ == '__main__':
    main()
