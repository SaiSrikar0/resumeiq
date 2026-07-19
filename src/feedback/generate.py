import os
import json
import pickle
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.classical_baseline import check_degree_match, build_features_optimized
from src.retrieval.vector_store import load_index
from src.models.embedding_model import load_model

def get_resume_by_id(resume_id, resumes_df):
    """Utility to retrieve a resume row by ID."""
    matches = resumes_df[resumes_df['ResumeID'] == resume_id]
    if matches.empty:
        raise ValueError(f"Resume ID {resume_id} not found.")
    return matches.iloc[0]

def retrieve_similar_resumes(resume_row, faiss_db, resumes_df, k=10):
    """Retrieves similar same-category resumes from the FAISS vector store."""
    resume_id = resume_row['ResumeID']
    category = resume_row['Category']
    
    # Query FAISS using the resume content
    # Filter by 'resume' type and same category
    query_text = f"Resume ID: {resume_id}\nCategory: {category}\nText: {resume_row['Text']}"
    
    # Search with metadata filter
    results = faiss_db.similarity_search(
        query_text,
        k=k + 1,  # Retrieve k+1 in case the query resume itself is returned
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
    """Retrieves relevant best practice snippets for the candidate's gaps."""
    missing = explain_obj.get('missing_skills', [])
    exp_gap = explain_obj.get('experience_gap', 0.0)
    edu_match = explain_obj.get('education_fit', {}).get('is_match', True)
    
    # Formulate search queries based on gaps
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
    
    # Retrieve from FAISS filtered by best_practice
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
    
    # 1. Retrieve similar same-category resumes
    similar_candidates = retrieve_similar_resumes(resume_row, faiss_db, resumes_df, k=15)
    
    # Score them against this JD to find higher-scoring ones
    higher_scoring_peers = []
    
    # Pre-transform TF-IDF representations of similar resumes and the current JD
    from sklearn.preprocessing import normalize
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
            
    # Sort peers by score descending
    higher_scoring_peers = sorted(higher_scoring_peers, key=lambda x: x['score'], reverse=True)[:3]
    
    # 2. Retrieve relevant best-practice snippets
    best_practices_retrieved = retrieve_best_practices(explain_obj, faiss_db, k=4)
    
    # 3. Generate Feedback Document
    if use_llm and llm_model is not None:
        # LLM based generation
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
        # Fallback Template-based generation
        report_lines = []
        report_lines.append(f"# ResumeIQ Feedback Report")
        report_lines.append(f"**Target Role:** {jd_row['Title']} ({category})")
        report_lines.append(f"**Candidate ID:** {resume_id} | **Fit Score:** {cand_score:.1f}%\n")
        
        report_lines.append("## Score & Explainability Summary")
        report_lines.append(explain_obj['explanation_text'] + "\n")
        
        report_lines.append("## Actionable Suggestions (Grounded in Gaps)")
        
        # Skill-related suggestions
        if missing_skills:
            report_lines.append("### Technical Skills Alignment")
            report_lines.append(f"*Gaps Addressed: Missing Skills ({', '.join(missing_skills[:5])})*")
            report_lines.append(f"**[Addresses Gap: Missing Skills]** Your resume is currently missing key skills listed in the job description: **{', '.join(missing_skills)}**.")
            
            # Find relevant snippets (we look for ones mentioning 'skills')
            skills_bps = [bp for bp in best_practices_retrieved if 'skill' in bp.lower() or 'technical' in bp.lower()]
            if not skills_bps:
                skills_bps = best_practices_retrieved[:2]
            for bp in skills_bps:
                report_lines.append(f"- {bp}")
            report_lines.append("")
            
        # Experience-related suggestions
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
            
        # Education suggestions
        if not edu_match:
            report_lines.append("### Education & Credentials")
            report_lines.append(f"*Gaps Addressed: Education Mismatch*")
            report_lines.append(f"**[Addresses Gap: Education Mismatch]** The job description requires a {jd_row['RequiredEducation']} degree, but your resume lists {resume_row['DegreeLevel']}.")
            
            edu_bps = [bp for bp in best_practices_retrieved if 'education' in bp.lower() or 'degree' in bp.lower() or 'certification' in bp.lower()]
            if not edu_bps:
                # Find any generic formatting / backup
                edu_bps = [bp for bp in best_practices_retrieved if 'certif' in bp.lower()]
                if not edu_bps:
                    edu_bps = [best_practices_retrieved[-1]]
            for bp in edu_bps:
                report_lines.append(f"- {bp}")
            report_lines.append("")
            
        # Formatting suggestions (general)
        report_lines.append("### General Formatting & Action Words")
        report_lines.append(f"**[Addresses Gap: General Scannability]** To maximize parsing rates and highlight your strengths:")
        fmt_bps = [bp for bp in best_practices_retrieved if 'verb' in bp.lower() or 'page' in bp.lower() or 'format' in bp.lower() or 'proofread' in bp.lower()]
        if not fmt_bps:
            fmt_bps = [best_practices_retrieved[0]]
        for bp in fmt_bps:
            report_lines.append(f"- {bp}")
        report_lines.append("")
        
        # Peer References
        report_lines.append("## Peer References")
        if higher_scoring_peers:
            report_lines.append("Here are details from similar candidates in your category who scored higher against this job description. Use their phrasing and skill highlights as references:")
            for p in higher_scoring_peers:
                skills_str = ", ".join(p['skills'][:5])
                report_lines.append(f"- **Candidate {p['id']}** (Fit Score: **{p['score']:.1f}%** | Experience: **{p['experience']:.1f} years**)")
                report_lines.append(f"  - *Skills Emphasized:* {skills_str}")
                if p['summary'] and len(p['summary']) > 5:
                    # Clean clean summary excerpt
                    clean_summary = p['summary'].replace('\n', ' ').strip()
                    if len(clean_summary) > 150:
                        clean_summary = clean_summary[:147] + "..."
                    report_lines.append(f"  - *Summary Excerpt:* \"{clean_summary}\"")
        else:
            report_lines.append("No higher-scoring peer resumes were identified for comparison.")
            
        feedback_text = "\n".join(report_lines)
        
    return feedback_text

def main():
    print("Testing Feedback Generation...")
    # Load resources
    from src.feedback.explain import load_baseline, compute_explainability_breakdown
    baseline_clf, baseline_type, importances = load_baseline()
    
    with open('src/models/artifacts/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        
    tokenizer, model = load_model()
    faiss_db = load_index(model, tokenizer)
    
    # Load datasets
    resumes_df = pd.read_parquet('data/processed/processed_resumes.parquet')
    jds_df = pd.read_json('data/processed/job_descriptions.jsonl', lines=True)
    
    res = resumes_df.iloc[0]
    jd = jds_df.iloc[0]
    
    explain_obj = compute_explainability_breakdown(res, jd, baseline_clf, baseline_type, importances, tfidf, model, tokenizer)
    feedback = generate_feedback_report(res, jd, explain_obj, faiss_db, resumes_df, baseline_clf, baseline_type, tfidf, use_llm=False)
    
    print("\nGenerated Feedback Report Preview:")
    print(feedback[:1000])
    
if __name__ == '__main__':
    main()
