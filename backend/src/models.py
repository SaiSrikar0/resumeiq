import os
import json
import pickle
import random
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import lightgbm as lgb

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'distilbert-base-uncased'

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
    res_id_to_idx = {res_id: i for i, res_id in enumerate(resumes_df['ResumeID'])}
    jd_id_to_idx = {jd_id: i for i, jd_id in enumerate(jds_df['JobDescriptionID'])}
    
    jds_by_cat = {}
    all_jds = []
    for _, jd_row in jds_df.iterrows():
        cat = jd_row['Category']
        jds_by_cat.setdefault(cat, []).append(jd_row)
        all_jds.append(jd_row)
        
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

# =====================================================================
# DEEP LEARNING (formerly embedding_model.py)
# =====================================================================

def load_model():
    print(f"Loading tokenizer and model {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.to(device)
    model.eval()
    return tokenizer, model

def get_embeddings_batched(texts, model, tokenizer, pooling='mean', batch_size=32, max_length=512):
    """Generates embeddings for a list of texts in batches."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
        attention_mask = inputs['attention_mask']  # [batch, seq_len]
        
        if pooling == 'cls':
            embs = last_hidden[:, 0, :]
        elif pooling == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            embs = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pooling {pooling}")
            
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)

def evaluate_pooling_strategies(resumes_df, jds_df, model, tokenizer, num_samples=100):
    """Compares 'cls' and 'mean' pooling strategies."""
    print("Evaluating pooling strategies...")
    random.seed(42)
    
    jds_by_cat = {}
    for _, jd_row in jds_df.iterrows():
        jds_by_cat.setdefault(jd_row['Category'], []).append(jd_row)
        
    same_pairs = []
    cross_pairs = []
    
    for _, res_row in resumes_df.iterrows():
        cat = res_row['Category']
        pos_jds = jds_by_cat.get(cat, [])
        if pos_jds:
            same_pairs.append((res_row['Text'], random.choice(pos_jds)['Description']))
            
        other_cats = [c for c in jds_by_cat if c != cat]
        if other_cats:
            neg_jd = random.choice(jds_by_cat[random.choice(other_cats)])
            cross_pairs.append((res_row['Text'], neg_jd['Description']))
            
    same_sample = random.sample(same_pairs, min(len(same_pairs), num_samples))
    cross_sample = random.sample(cross_pairs, min(len(cross_pairs), num_samples))
    
    results = {}
    for pooling in ['cls', 'mean']:
        print(f"Testing {pooling} pooling...")
        all_texts = list(set([p[0] for p in same_sample] + [p[1] for p in same_sample] +
                             [p[0] for p in cross_sample] + [p[1] for p in cross_sample]))
        
        embs_dict = {}
        embs = get_embeddings_batched(all_texts, model, tokenizer, pooling=pooling, batch_size=32)
        for text, emb in zip(all_texts, embs):
            embs_dict[text] = emb
            
        pos_sims = []
        for res_text, jd_text in same_sample:
            r_emb = embs_dict[res_text].reshape(1, -1)
            j_emb = embs_dict[jd_text].reshape(1, -1)
            pos_sims.append(cosine_similarity(r_emb, j_emb)[0, 0])
            
        neg_sims = []
        for res_text, jd_text in cross_sample:
            r_emb = embs_dict[res_text].reshape(1, -1)
            j_emb = embs_dict[jd_text].reshape(1, -1)
            neg_sims.append(cosine_similarity(r_emb, j_emb)[0, 0])
            
        mean_pos = np.mean(pos_sims)
        mean_neg = np.mean(neg_sims)
        separation = mean_pos - mean_neg
        
        results[pooling] = {
            'mean_positive_sim': float(mean_pos),
            'mean_negative_sim': float(mean_neg),
            'separation': float(separation)
        }
        print(f"Pooling '{pooling}' -> Pos Sim: {mean_pos:.4f}, Neg Sim: {mean_neg:.4f}, Sep: {separation:.4f}")
        
    best_pooling = 'mean' if results['mean']['separation'] >= results['cls']['separation'] else 'cls'
    print(f"Selected pooling: {best_pooling}")
    return best_pooling, results

def get_attention_explainability(resume_text, jd_embedding, model, tokenizer, max_length=512):
    """Extracts last-layer attention weights."""
    inputs = tokenizer(resume_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    last_hidden = outputs.last_hidden_state[0]
    attentions = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
    cls_attention = attentions[:, 0, :].mean(dim=0).cpu().numpy()
    token_embeddings = last_hidden.cpu().numpy()
    
    norm_token_embs = token_embeddings / np.linalg.norm(token_embeddings, axis=1, keepdims=True)
    norm_jd_emb = jd_embedding / np.linalg.norm(jd_embedding)
    similarities = np.dot(norm_token_embs, norm_jd_emb)
    
    influences = cls_attention * np.maximum(0.0, similarities)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    merged_words = []
    merged_influences = []
    current_word = ""
    current_influence = 0.0
    
    for token, influence in zip(tokens, influences):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        if token.startswith("##"):
            current_word += token[2:]
            current_influence += influence
        else:
            if current_word:
                merged_words.append(current_word)
                merged_influences.append(current_influence)
            current_word = token
            current_influence = influence
            
    if current_word:
        merged_words.append(current_word)
        merged_influences.append(current_influence)
        
    word_scores = {}
    for word, score in zip(merged_words, merged_influences):
        word_clean = word.strip().lower()
        if len(word_clean) <= 2 or word_clean in ['the', 'and', 'for', 'with', 'that', 'this', 'from']:
            continue
        word_scores[word] = word_scores.get(word, 0.0) + score
        
    top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_words_list = [w[0] for w in top_words]
    explain_note = f"The model focused most heavily on '{', '.join(top_words_list)}' in the resume when matching with the JD."
    
    return {
        'top_tokens': top_words_list,
        'explainability_note': explain_note
    }

def main():
    print("=" * 60)
    print("ResumeIQ: Models Training & Embedding Generation")
    print("=" * 60)
    
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
    
    res_vecs = tfidf_vectorizer.transform(resumes_df['Text'].fillna(''))
    jd_vecs = tfidf_vectorizer.transform(jds_df['Description'].fillna(''))
    res_vecs_norm = normalize(res_vecs, norm='l2', axis=1)
    jd_vecs_norm = normalize(jd_vecs, norm='l2', axis=1)
    
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print("TF-IDF Vectorizer saved to artifacts/tfidf_vectorizer.pkl")
    
    # 3. Construct paired dataset
    paired_df = prepare_dataset(resumes_df, jds_df, res_vecs_norm, jd_vecs_norm)
    features = ['skill_overlap_ratio', 'experience_gap', 'degree_match', 'tfidf_similarity']
    
    # 4. Compare XGBoost and LightGBM
    print("Evaluating XGBoost...")
    xgb_metrics = run_cross_validation(paired_df, features, model_type='xgboost')
    print("Evaluating LightGBM...")
    lgb_metrics = run_cross_validation(paired_df, features, model_type='lightgbm')
    
    best_model_type = 'xgboost' if xgb_metrics['auc'] >= lgb_metrics['auc'] else 'lightgbm'
    print(f"\nBest baseline model: {best_model_type.upper()}")
    
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
        model.save_model('artifacts/baseline_xgb.json')
        
        importance_scores = model.get_booster().get_score(importance_type='gain')
        feature_importances = {}
        for i, feat in enumerate(features):
            val = importance_scores.get(feat, importance_scores.get(f'f{i}', 0.0))
            feature_importances[feat] = float(val)
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
        model.booster_.save_model('artifacts/baseline_lgb.txt')
        
        importances = model.booster_.feature_importance(importance_type='gain')
        feature_importances = {feat: float(imp) for feat, imp in zip(features, importances)}
        total_importance = sum(feature_importances.values())
        if total_importance > 0:
            feature_importances = {k: v / total_importance for k, v in feature_importances.items()}
            
    metadata = {
        'best_model_type': best_model_type,
        'xgb_cv_metrics': xgb_metrics,
        'lgb_cv_metrics': lgb_metrics,
        'feature_importances': feature_importances
    }
    with open('artifacts/baseline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # 6. Deep Learning Tokenizer & Embeddings
    tokenizer, dl_model = load_model()
    best_pooling, pooling_eval_results = evaluate_pooling_strategies(resumes_df, jds_df, dl_model, tokenizer, num_samples=100)
    
    # Save embeddings
    print("Generating embeddings for all resumes...")
    resume_texts = resumes_df['Text'].fillna('').tolist()
    resume_embs = get_embeddings_batched(resume_texts, dl_model, tokenizer, pooling=best_pooling, batch_size=32)
    np.save('artifacts/resume_embeddings.npy', resume_embs)
    
    resume_ids = resumes_df['ResumeID'].tolist()
    with open('artifacts/resume_ids.json', 'w') as f:
        json.dump(resume_ids, f)
        
    # Agreement
    print("Computing ranking agreement...")
    sample_pairs = []
    res_list = resumes_df.to_dict('records')
    jd_list = jds_df.to_dict('records')
    for _ in range(200):
        sample_pairs.append((random.choice(res_list), random.choice(jd_list)))
        
    baseline_scores = []
    embedding_scores = []
    res_texts = [p[0]['Text'] for p in sample_pairs]
    jd_texts = [p[1]['Description'] for p in sample_pairs]
    
    res_tfidfs = normalize(tfidf_vectorizer.transform(res_texts), norm='l2', axis=1)
    jd_tfidfs = normalize(tfidf_vectorizer.transform(jd_texts), norm='l2', axis=1)
    
    # Re-instantiate model for prediction
    if best_model_type == 'xgboost':
        clf = xgb.XGBClassifier()
        clf.load_model('artifacts/baseline_xgb.json')
    else:
        clf = lgb.Booster(model_file='artifacts/baseline_lgb.txt')
        
    for i, (res, jd) in enumerate(sample_pairs):
        cand_skills = set(s.lower() for s in res.get('ExtractedSkills', []))
        req_skills = set(s.lower() for s in jd.get('RequiredSkills', []))
        skill_overlap = len(cand_skills & req_skills) / len(req_skills) if req_skills else 0.0
        
        cand_exp = float(res.get('YearsOfExperience', 0.0))
        req_exp = float(jd.get('RequiredExperience', 0.0))
        exp_gap = req_exp - cand_exp
        
        degree_match = check_degree_match(res.get('DegreeLevel', 'None'), jd.get('RequiredEducation', 'None'))
        tfidf_sim = float((res_tfidfs[i] * jd_tfidfs[i].T)[0, 0])
        
        feats_df = pd.DataFrame([{
            'skill_overlap_ratio': skill_overlap,
            'experience_gap': exp_gap,
            'degree_match': degree_match,
            'tfidf_similarity': tfidf_sim
        }])
        
        if best_model_type == 'xgboost':
            score = float(clf.predict_proba(feats_df)[:, 1][0] * 100.0)
        else:
            score = float(clf.predict(feats_df)[0] * 100.0)
        baseline_scores.append(score)
        
        r_emb = get_embeddings_batched([res['Text']], dl_model, tokenizer, pooling=best_pooling, batch_size=1)[0]
        j_emb = get_embeddings_batched([jd['Description']], dl_model, tokenizer, pooling=best_pooling, batch_size=1)[0]
        cos_sim = float(np.dot(r_emb, j_emb) / (np.linalg.norm(r_emb) * np.linalg.norm(j_emb)))
        embedding_scores.append(cos_sim)
        
    corr, p_val = spearmanr(baseline_scores, embedding_scores)
    emb_metadata = {
        'best_pooling': best_pooling,
        'pooling_eval_results': pooling_eval_results,
        'spearman_correlation': float(corr),
        'spearman_p_value': float(p_val)
    }
    with open('artifacts/embedding_metadata.json', 'w') as f:
        json.dump(emb_metadata, f, indent=4)
        
    print(f"Ranking agreement (Spearman Rank Correlation): {corr:.4f}")
    print("Everything trained and saved to top-level artifacts/ successfully!")

if __name__ == '__main__':
    main()
