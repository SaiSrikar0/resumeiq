import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import normalize
from src.models import check_degree_match, build_features_optimized, load_model, get_embeddings_batched, get_attention_explainability
from src.retrieval import load_index

def load_baseline():
    """Loads the best-performing classical ML model and its feature importances."""
    possible_dirs = [
        "artifacts",
        "backend/artifacts",
        os.path.join(os.path.dirname(__file__), "..", "artifacts"),
    ]
    base_dir = None
    for pd_dir in possible_dirs:
        if os.path.exists(os.path.join(pd_dir, "baseline_metadata.json")):
            base_dir = pd_dir
            break

    if not base_dir:
        raise FileNotFoundError("Baseline metadata not found. Run prepare_data.py and models training first.")

    metadata_path = os.path.join(base_dir, "baseline_metadata.json")
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
        
    best_type = meta['best_model_type']
    importances = meta['feature_importances']
    
    if best_type == 'xgboost':
        import xgboost as xgb
        clf = xgb.XGBClassifier()
        clf.load_model(os.path.join(base_dir, 'baseline_xgb.json'))
    else:
        import lightgbm as lgb
        clf = lgb.Booster(model_file=os.path.join(base_dir, 'baseline_lgb.txt'))
        
    return clf, best_type, importances

def get_fit_score(clf, model_type, features_dict) -> float:
    """Computes baseline model prediction (probability of match scaled to 0-100)."""
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
    """Generates a plain-language explanation of why a candidate fits or does not fit a JD."""
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
    sentences.append(
        f"ResumeIQ calculated a fit score of {fit_score:.1f}% for this role, driven primarily by "
        f"a skill overlap of {overlap * 100:.0f}% and semantic text similarity."
    )
    
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
    
    if exp_gap > 0:
        sentences.append(
            f"There is an experience gap of {exp_gap:.1f} years: the candidate has {cand_exp:.1f} years, "
            f"while the role requires {req_exp:.1f} years."
        )
    else:
        sentences.append(
            f"The candidate's {cand_exp:.1f} years of experience successfully meets the required {req_exp:.1f} years."
        )
        
    edu_status = "meets or exceeds" if edu_match else "does not meet"
    attention_str = ", ".join(top_tokens[:3])
    sentences.append(
        f"The candidate's education ({cand_edu}) {edu_status} the required {req_edu}. "
        f"Additionally, the neural model's attention highlights that terminology surrounding "
        f"'{attention_str}' in the resume was highly influential during matching."
    )
    
    return " ".join(sentences)

def compute_explainability_breakdown(resume_row, jd_row, baseline_clf, baseline_type, importances, tfidf_vectorizer, embedding_model, embedding_tokenizer):
    """Computes a structured explainability object combining ML features and transformer attention."""
    cand_skills = resume_row.get('ExtractedSkills', [])
    req_skills = jd_row.get('RequiredSkills', [])
    
    cand_skills_set = set(s.lower() for s in cand_skills)
    req_skills_set = set(s.lower() for s in req_skills)
    
    matched_skills = sorted(list(set(s for s in req_skills if s.lower() in cand_skills_set)))
    missing_skills = sorted(list(set(s for s in req_skills if s.lower() not in cand_skills_set)))
    
    skill_overlap = len(cand_skills_set & req_skills_set) / len(req_skills_set) if req_skills_set else 0.0
    
    cand_exp = float(resume_row.get('YearsOfExperience', 0.0))
    req_exp = float(jd_row.get('RequiredExperience', 0.0))
    exp_gap = req_exp - cand_exp
    
    cand_edu = resume_row.get('DegreeLevel', 'None')
    req_edu = jd_row.get('RequiredEducation', 'None')
    edu_match = check_degree_match(cand_edu, req_edu) == 1.0
    
    res_text = resume_row.get('Text', '')
    jd_text = jd_row.get('Description', '')
    
    res_vec = tfidf_vectorizer.transform([res_text])
    jd_vec = tfidf_vectorizer.transform([jd_text])
    res_norm = normalize(res_vec, norm='l2', axis=1)
    jd_norm = normalize(jd_vec, norm='l2', axis=1)
    tfidf_sim = float((res_norm * jd_norm.T)[0, 0])
    
    feats = {
        'skill_overlap_ratio': skill_overlap,
        'experience_gap': exp_gap,
        'degree_match': 1.0 if edu_match else 0.0,
        'tfidf_similarity': tfidf_sim
    }
    model_score = get_fit_score(baseline_clf, baseline_type, feats)
    
    skill_score = skill_overlap
    if exp_gap <= 0:
        exp_score = 1.0
    else:
        exp_score = max(0.0, 1.0 - (exp_gap / req_exp)) if req_exp > 0 else 0.0
        
    deg_score = 1.0 if edu_match else 0.0
    semantic_score = model_score / 100.0
    
    fit_score = 0.40 * skill_score + 0.30 * exp_score + 0.15 * deg_score + 0.15 * semantic_score
    fit_score = float(fit_score * 100.0)
    
    best_pooling = 'mean'
    jd_emb = get_embeddings_batched([jd_text], embedding_model, embedding_tokenizer, pooling=best_pooling, batch_size=1)[0]
    attn_explain = get_attention_explainability(res_text, jd_emb, embedding_model, embedding_tokenizer)
    
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
    
    explain_obj['explanation_text'] = generate_explanation(explain_obj)
    return explain_obj

# =====================================================================
# RAG FEEDBACK REPORT (formerly generate.py)
# =====================================================================

def get_resume_by_id(resume_id, resumes_df):
    matches = resumes_df[resumes_df['ResumeID'] == resume_id]
    if matches.empty:
        raise ValueError(f"Resume ID {resume_id} not found.")
    return matches.iloc[0]

def retrieve_similar_resumes(resume_row, faiss_db, resumes_df, k=10):
    resume_id = resume_row['ResumeID']
    category = resume_row['Category']
    query_text = f"Resume ID: {resume_id}\nCategory: {category}\nText: {resume_row['Text']}"
    
    results = faiss_db.similarity_search(
        query_text,
        k=k + 1,
        filter={'type': 'resume', 'category': category},
        fetch_k=3500
    )
    
    similar_resumes = []
    for doc in results:
        res_id = doc.metadata['id']
        if res_id == resume_id:
            continue
        try:
            row = get_resume_by_id(res_id, resumes_df)
            similar_resumes.append(row)
        except ValueError:
            continue
            
    return similar_resumes[:k]

def retrieve_best_practices(explain_obj, faiss_db, k=4):
    missing = explain_obj.get('missing_skills', [])
    exp_gap = explain_obj.get('experience_gap', 0.0)
    edu_match = explain_obj.get('education_fit', {}).get('is_match', True)
    
    queries = []
    if missing:
        queries.append(f"missing technical skills: {', '.join(missing[:4])}")
    if exp_gap > 0:
        queries.append(f"experience gap of {exp_gap:.1f} years, career progression, projects")
    if not edu_match:
        queries.append("degree mismatch, certifications, equivalent education")
        
    if not queries:
        queries.append("general resume formatting, active language, quantifying impact")
        
    query_str = " | ".join(queries)
    results = faiss_db.similarity_search(
        query_str,
        k=k,
        filter={'type': 'best_practice'},
        fetch_k=3500
    )
    return [doc.page_content for doc in results]

def generate_feedback_report(resume_row, jd_row, explain_obj, faiss_db, resumes_df, baseline_clf, baseline_type, tfidf_vectorizer, use_llm=False, llm_model=None):
    """Orchestrates retrieval and generates grounded feedback text."""
    resume_id = resume_row['ResumeID']
    jd_id = jd_row['JobDescriptionID']
    category = resume_row['Category']
    
    cand_score = explain_obj['baseline_fit_score']
    missing_skills = explain_obj['missing_skills']
    exp_gap = explain_obj['experience_gap']
    edu_match = explain_obj['education_fit']['is_match']
    
    similar_candidates = retrieve_similar_resumes(resume_row, faiss_db, resumes_df, k=15)
    higher_scoring_peers = []
    
    jd_tfidf = normalize(tfidf_vectorizer.transform([jd_row['Description']]), norm='l2', axis=1)
    
    for peer in similar_candidates:
        peer_skills = set(s.lower() for s in peer.get('ExtractedSkills', []))
        req_skills = set(s.lower() for s in jd_row.get('RequiredSkills', []))
        peer_overlap = len(peer_skills & req_skills) / len(req_skills) if req_skills else 0.0
        
        peer_exp = float(peer.get('YearsOfExperience', 0.0))
        peer_exp_gap = float(jd_row.get('RequiredExperience', 0.0)) - peer_exp
        peer_degree_match = check_degree_match(peer.get('DegreeLevel', 'None'), jd_row.get('RequiredEducation', 'None'))
        
        peer_tfidf = normalize(tfidf_vectorizer.transform([peer['Text']]), norm='l2', axis=1)
        peer_tfidf_sim = float((peer_tfidf * jd_tfidf.T)[0, 0])
        
        features_list = [peer_overlap, peer_exp_gap, peer_degree_match, peer_tfidf_sim]
        df = pd.DataFrame([features_list], columns=[
            'skill_overlap_ratio', 'experience_gap', 'degree_match', 'tfidf_similarity'
        ])
        
        if baseline_type == 'xgboost':
            peer_score = float(baseline_clf.predict_proba(df)[:, 1][0] * 100.0)
        else:
            peer_score = float(baseline_clf.predict(df)[0] * 100.0)
            
        if peer_score > cand_score:
            higher_scoring_peers.append({
                'id': peer['ResumeID'],
                'score': peer_score,
                'skills': peer.get('ExtractedSkills', []),
                'experience': peer_exp,
                'summary': peer.get('Summary', '')
            })
            
    higher_scoring_peers = sorted(higher_scoring_peers, key=lambda x: x['score'], reverse=True)[:3]
    best_practices_retrieved = retrieve_best_practices(explain_obj, faiss_db, k=4)
    
    if use_llm and llm_model is not None:
        prompt = f"""
You are ResumeIQ, an expert career advisor.
Given the candidate's metrics, explainability, and retrieved support documents, draft a professional, grounded feedback report.

Candidate Resume ID: {resume_id}
Target Job Title: {jd_row['Title']}
Fit Score: {cand_score:.1f}%

Explainability Summary:
{explain_obj['explanation_text']}

Gaps Detected:
- Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}
- Experience Gap: {exp_gap:.1f} years (Requires {jd_row['RequiredExperience']}, Candidate has {resume_row['YearsOfExperience']})
- Education Match: {'Yes' if edu_match else 'No (Requires ' + jd_row['RequiredEducation'] + ', Candidate has ' + resume_row['DegreeLevel'] + ')'}

Retrieved Best Practice Suggestions:
{chr(10).join(f'- {bp}' for bp in best_practices_retrieved)}

Higher-Scoring Peers in same category:
{chr(10).join(f'- Candidate {p["id"]} (Score: {p["score"]:.1f}%): emphasizes skills {p["skills"][:5]}' for p in higher_scoring_peers)}

Write a highly structured feedback document containing:
1. Score & Explainability Summary
2. Actionable Suggestions (Grounded in Gaps) - every suggestion must explicitly cite which specific gap it addresses (e.g., "[Addresses Gap: Missing Skills]") using the retrieved best practices.
3. Peer References - explain what keywords or skills the higher-scoring peers emphasized.
"""
        response = llm_model.invoke(prompt)
        feedback_text = response.content if hasattr(response, 'content') else str(response)
        
    else:
        # Template-based generation
        report_lines = []
        report_lines.append(f"# ResumeIQ Feedback Report")
        report_lines.append(f"**Target Role:** {jd_row['Title']} ({category})")
        report_lines.append(f"**Candidate ID:** {resume_id} | **Fit Score:** {cand_score:.1f}%\n")
        
        report_lines.append("## Score & Explainability Summary")
        report_lines.append(explain_obj['explanation_text'] + "\n")
        report_lines.append("## Actionable Suggestions (Grounded in Gaps)")
        
        if missing_skills:
            report_lines.append("### Technical Skills Alignment")
            report_lines.append(f"*Gaps Addressed: Missing Skills ({', '.join(missing_skills[:5])})*")
            report_lines.append(f"**[Addresses Gap: Missing Skills]** Your resume is currently missing key skills listed in the job description: **{', '.join(missing_skills)}**.")
            
            skills_bps = [bp for bp in best_practices_retrieved if 'skill' in bp.lower() or 'technical' in bp.lower()]
            if not skills_bps:
                skills_bps = best_practices_retrieved[:2]
            for bp in skills_bps:
                report_lines.append(f"- {bp}")
            report_lines.append("")
            
        if exp_gap > 0:
            report_lines.append("### Experience & Impact Presentation")
            report_lines.append(f"*Gaps Addressed: Experience Gap ({exp_gap:.1f} years)*")
            report_lines.append(f"**[Addresses Gap: Experience Gap ({exp_gap:.1f} years)]** You have {resume_row['YearsOfExperience']:.1f} years of experience, which is short of the required {jd_row['RequiredExperience']:.1f} years.")
            
            exp_bps = [bp for bp in best_practices_retrieved if 'experience' in bp.lower() or 'gap' in bp.lower() or 'quantify' in bp.lower() or 'impact' in bp.lower()]
            if not exp_bps:
                exp_bps = best_practices_retrieved[1:3]
            for bp in exp_bps:
                report_lines.append(f"- {bp}")
            report_lines.append("")
            
        if not edu_match:
            report_lines.append("### Education & Credentials")
            report_lines.append(f"*Gaps Addressed: Education Mismatch*")
            report_lines.append(f"**[Addresses Gap: Education Mismatch]** The job description requires a {jd_row['RequiredEducation']} degree, but your resume lists {resume_row['DegreeLevel']}.")
            
            edu_bps = [bp for bp in best_practices_retrieved if 'education' in bp.lower() or 'degree' in bp.lower() or 'certification' in bp.lower()]
            if not edu_bps:
                edu_bps = [bp for bp in best_practices_retrieved if 'certif' in bp.lower()]
                if not edu_bps:
                    edu_bps = [best_practices_retrieved[-1]]
            for bp in edu_bps:
                report_lines.append(f"- {bp}")
            report_lines.append("")
            
        report_lines.append("### General Formatting & Action Words")
        report_lines.append(f"**[Addresses Gap: General Scannability]** To maximize parsing rates and highlight your strengths:")
        fmt_bps = [bp for bp in best_practices_retrieved if 'verb' in bp.lower() or 'page' in bp.lower() or 'format' in bp.lower() or 'proofread' in bp.lower()]
        if not fmt_bps:
            fmt_bps = [best_practices_retrieved[0]]
        for bp in fmt_bps:
            report_lines.append(f"- {bp}")
        report_lines.append("")
        
        report_lines.append("## Peer References")
        if higher_scoring_peers:
            report_lines.append("Here are details from similar candidates in your category who scored higher against this job description. Use their phrasing and skill highlights as references:")
            for p in higher_scoring_peers:
                skills_str = ", ".join(p['skills'][:5])
                report_lines.append(f"- **Candidate {p['id']}** (Fit Score: **{p['score']:.1f}%** | Experience: **{p['experience']:.1f} years**)")
                report_lines.append(f"  - *Skills Emphasized:* {skills_str}")
                if p['summary'] and len(p['summary']) > 5:
                    clean_summary = p['summary'].replace('\n', ' ').strip()
                    if len(clean_summary) > 150:
                        clean_summary = clean_summary[:147] + "..."
                    report_lines.append(f"  - *Summary Excerpt:* \"{clean_summary}\"")
        else:
            report_lines.append("No higher-scoring peer resumes were identified for comparison.")
            
        feedback_text = "\n".join(report_lines)
        
    return feedback_text
