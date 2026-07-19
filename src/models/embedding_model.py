import os
import json
import pickle
import random
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
import sys
from pathlib import Path
# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'distilbert-base-uncased'

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
    """
    Compares 'cls' and 'mean' pooling strategies.
    Measures the separation between same-category (positive) and cross-category (negative) pairs.
    """
    print("Evaluating pooling strategies...")
    # Select sample pairs
    random.seed(42)
    
    # Pre-select some same-category and cross-category pairs
    jds_by_cat = {}
    for _, jd_row in jds_df.iterrows():
        jds_by_cat.setdefault(jd_row['Category'], []).append(jd_row)
        
    same_pairs = []
    cross_pairs = []
    
    # Find matching resume/jd pairs
    for _, res_row in resumes_df.iterrows():
        cat = res_row['Category']
        pos_jds = jds_by_cat.get(cat, [])
        if pos_jds:
            same_pairs.append((res_row['Text'], random.choice(pos_jds)['Description']))
            
        # Cross category
        other_cats = [c for c in jds_by_cat if c != cat]
        if other_cats:
            neg_jd = random.choice(jds_by_cat[random.choice(other_cats)])
            cross_pairs.append((res_row['Text'], neg_jd['Description']))
            
    # Sample down for speed of evaluation
    same_sample = random.sample(same_pairs, min(len(same_pairs), num_samples))
    cross_sample = random.sample(cross_pairs, min(len(cross_pairs), num_samples))
    
    results = {}
    for pooling in ['cls', 'mean']:
        print(f"Testing {pooling} pooling...")
        
        # We can extract all unique texts to embed to save computation
        all_texts = list(set([p[0] for p in same_sample] + [p[1] for p in same_sample] +
                             [p[0] for p in cross_sample] + [p[1] for p in cross_sample]))
        
        embs_dict = {}
        # Embed in batches
        embs = get_embeddings_batched(all_texts, model, tokenizer, pooling=pooling, batch_size=32)
        for text, emb in zip(all_texts, embs):
            embs_dict[text] = emb
            
        # Compute similarities
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
    """
    Extracts last-layer attention weights to surface which resume words most influenced the embedding relative to the JD.
    """
    inputs = tokenizer(resume_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # Get last-hidden states (representations) [batch, seq_len, hidden]
    last_hidden = outputs.last_hidden_state[0]
    
    # Get attention weights of last layer [batch, num_heads, seq_len, seq_len]
    # For DistilBERT, model outputs.attentions is a tuple of length 6
    attentions = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
    
    # Average attention from [CLS] (token index 0) across all heads
    # shape: [seq_len]
    cls_attention = attentions[:, 0, :].mean(dim=0).cpu().numpy()
    
    # Convert token representations to CPU numpy
    token_embeddings = last_hidden.cpu().numpy()  # [seq_len, hidden]
    
    # L2 normalize token embeddings and JD embedding for cosine similarity
    norm_token_embs = token_embeddings / np.linalg.norm(token_embeddings, axis=1, keepdims=True)
    norm_jd_emb = jd_embedding / np.linalg.norm(jd_embedding)
    
    # Calculate similarity of each token to the overall JD representation
    similarities = np.dot(norm_token_embs, norm_jd_emb)  # [seq_len]
    
    # Compute combined influence score: attention weight * similarity
    # We clip similarity at 0 so tokens negatively correlated with the JD don't get high negative attention
    influences = cls_attention * np.maximum(0.0, similarities)
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Merge WordPieces
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
        
    # Group by word and sum influences (in case a word appears multiple times)
    word_scores = {}
    for word, score in zip(merged_words, merged_influences):
        # Filter out very short/noise words or common English stop words
        word_clean = word.strip().lower()
        if len(word_clean) <= 2 or word_clean in ['the', 'and', 'for', 'with', 'that', 'this', 'from']:
            continue
        word_scores[word] = word_scores.get(word, 0.0) + score
        
    # Get top 5 words
    top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_words_list = [w[0] for w in top_words]
    
    explain_note = f"The model focused most heavily on '{', '.join(top_words_list)}' in the resume when matching with the JD."
    
    return {
        'top_tokens': top_words_list,
        'explainability_note': explain_note
    }

def main():
    # 1. Load tokenizer and model
    tokenizer, model = load_model()
    
    # 2. Load resumes and JDs
    print("Loading datasets...")
    resumes_df = pd.read_parquet('data/processed/processed_resumes.parquet')
    jds_df = pd.read_json('data/processed/job_descriptions.jsonl', lines=True)
    
    # 3. Evaluate and select pooling strategy
    best_pooling, pooling_eval_results = evaluate_pooling_strategies(resumes_df, jds_df, model, tokenizer, num_samples=100)
    
    # 4. Generate and save embeddings for all resumes
    embedding_path = 'src/models/artifacts/resume_embeddings.npy'
    ids_path = 'src/models/artifacts/resume_ids.json'
    if os.path.exists(embedding_path) and os.path.exists(ids_path):
        print("Loading cached resume embeddings from artifacts...")
        resume_embs = np.load(embedding_path)
        with open(ids_path, 'r') as f:
            resume_ids = json.load(f)
    else:
        print("Generating embeddings for all resumes...")
        resume_texts = resumes_df['Text'].fillna('').tolist()
        resume_embs = get_embeddings_batched(resume_texts, model, tokenizer, pooling=best_pooling, batch_size=32)
        
        os.makedirs('src/models/artifacts', exist_ok=True)
        np.save(embedding_path, resume_embs)
        
        resume_ids = resumes_df['ResumeID'].tolist()
        with open(ids_path, 'w') as f:
            json.dump(resume_ids, f)
        print("Saved resume embeddings and IDs to artifacts.")
    
    # 5. Compute agreement with baseline LightGBM/XGBoost on a sample
    print("Computing ranking agreement with classical ML baseline...")
    # Load baseline model and metadata
    with open('src/models/artifacts/baseline_metadata.json', 'r') as f:
        meta = json.load(f)
        
    best_baseline = meta['best_model_type']
    print(f"Active baseline model: {best_baseline}")
    
    # Load TF-IDF vectorizer
    with open('src/models/artifacts/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        
    # Import baseline features logic
    from src.models.classical_baseline import check_degree_match, build_features_optimized
    
    # Construct feature matrix for 200 random pairs
    random.seed(99)
    sample_pairs = []
    
    # Just draw random resumes and JDs
    res_list = resumes_df.to_dict('records')
    jd_list = jds_df.to_dict('records')
    
    for _ in range(200):
        res = random.choice(res_list)
        jd = random.choice(jd_list)
        sample_pairs.append((res, jd))
        
    # Predict with LightGBM/XGBoost baseline
    import xgboost as xgb
    import lightgbm as lgb
    
    if best_baseline == 'xgboost':
        clf = xgb.XGBClassifier()
        clf.load_model('src/models/artifacts/baseline_xgb.json')
    else:
        clf = lgb.Booster(model_file='src/models/artifacts/baseline_lgb.txt')
        
    baseline_scores = []
    embedding_scores = []
    
    # Pre-transform TF-IDF for speed
    res_texts = [p[0]['Text'] for p in sample_pairs]
    jd_texts = [p[1]['Description'] for p in sample_pairs]
    
    res_tfidfs = normalize(tfidf.transform(res_texts), norm='l2', axis=1)
    jd_tfidfs = normalize(tfidf.transform(jd_texts), norm='l2', axis=1)
    
    for i, (res, jd) in enumerate(sample_pairs):
        # Baseline features
        # Create dummy rows
        res_row = res
        jd_row = jd
        
        # Call features
        cand_skills = set(s.lower() for s in res_row.get('ExtractedSkills', []))
        req_skills = set(s.lower() for s in jd_row.get('RequiredSkills', []))
        skill_overlap = len(cand_skills & req_skills) / len(req_skills) if req_skills else 0.0
        
        cand_exp = float(res_row.get('YearsOfExperience', 0.0))
        req_exp = float(jd_row.get('RequiredExperience', 0.0))
        exp_gap = req_exp - cand_exp
        
        degree_match = check_degree_match(res_row.get('DegreeLevel', 'None'), jd_row.get('RequiredEducation', 'None'))
        tfidf_sim = float((res_tfidfs[i] * jd_tfidfs[i].T)[0, 0])
        
        feats_df = pd.DataFrame([{
            'skill_overlap_ratio': skill_overlap,
            'experience_gap': exp_gap,
            'degree_match': degree_match,
            'tfidf_similarity': tfidf_sim
        }])
        
        if best_baseline == 'xgboost':
            score = float(clf.predict_proba(feats_df)[:, 1][0] * 100.0)
        else:
            # lightgbm predict takes numpy array or df
            score = float(clf.predict(feats_df)[0] * 100.0)
        baseline_scores.append(score)
        
        # Embedding cosine similarity
        r_emb = get_embeddings_batched([res['Text']], model, tokenizer, pooling=best_pooling, batch_size=1)[0]
        j_emb = get_embeddings_batched([jd['Description']], model, tokenizer, pooling=best_pooling, batch_size=1)[0]
        
        # Cosine Similarity
        cos_sim = float(np.dot(r_emb, j_emb) / (np.linalg.norm(r_emb) * np.linalg.norm(j_emb)))
        embedding_scores.append(cos_sim)
        
    # Spearman rank correlation
    corr, p_val = spearmanr(baseline_scores, embedding_scores)
    print(f"Ranking agreement (Spearman Rank Correlation): {corr:.4f} (p-value: {p_val:.2e})")
    
    # Save embedding model metadata
    emb_metadata = {
        'best_pooling': best_pooling,
        'pooling_eval_results': pooling_eval_results,
        'spearman_correlation': float(corr),
        'spearman_p_value': float(p_val)
    }
    with open('src/models/artifacts/embedding_metadata.json', 'w') as f:
        json.dump(emb_metadata, f, indent=4)
    print("Embedding metadata saved to src/models/artifacts/embedding_metadata.json")

if __name__ == '__main__':
    main()
