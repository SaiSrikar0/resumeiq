import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Lazy load helpers to keep imports fast
_data_cache = {}

def _load_data():
    if 'resumes_df' not in _data_cache:
        _data_cache['resumes_df'] = pd.read_parquet('data/processed_resumes.parquet')
    if 'jds_df' not in _data_cache:
        _data_cache['jds_df'] = pd.read_json('data/job_descriptions.jsonl', lines=True)
    return _data_cache['resumes_df'], _data_cache['jds_df']

def _load_models():
    if 'baseline_clf' not in _data_cache:
        from src.feedback.explain import load_baseline
        clf, b_type, importances = load_baseline()
        _data_cache['baseline_clf'] = clf
        _data_cache['baseline_type'] = b_type
        _data_cache['importances'] = importances
        
    if 'tfidf' not in _data_cache:
        with open('src/models/artifacts/tfidf_vectorizer.pkl', 'rb') as f:
            _data_cache['tfidf'] = pickle.load(f)
            
    if 'faiss_db' not in _data_cache:
        from src.retrieval.vector_store import load_index
        _data_cache['faiss_db'] = load_index()
        
    return (_data_cache['baseline_clf'], _data_cache['baseline_type'], 
            _data_cache['importances'], _data_cache['tfidf'], _data_cache['faiss_db'])

@tool
def get_score_breakdown(resume_id: str, jd_id: str) -> str:
    """
    Retrieves the fit score, experience gap, degree match, and skill overlap ratio for a resume and job description pair.
    Use this tool whenever the user asks for their score or a breakdown of how they matched.
    """
    try:
        resumes_df, jds_df = _load_data()
        baseline_clf, baseline_type, importances, tfidf, faiss_db = _load_models()
        
        # Look up resume and JD
        res_matches = resumes_df[resumes_df['ResumeID'] == resume_id]
        jd_matches = jds_df[jds_df['JobDescriptionID'] == jd_id]
        
        if res_matches.empty:
            return f"Error: Resume ID {resume_id} not found."
        if jd_matches.empty:
            return f"Error: Job Description ID {jd_id} not found."
            
        res_row = res_matches.iloc[0]
        jd_row = jd_matches.iloc[0]
        
        from src.feedback.explain import compute_explainability_breakdown
        breakdown = compute_explainability_breakdown(
            res_row, jd_row, baseline_clf, baseline_type, importances, tfidf, 
            faiss_db.embeddings.model, faiss_db.embeddings.tokenizer
        )
        
        edu_status = "meets or exceeds required" if breakdown['education_fit']['is_match'] else "does not meet required"
        
        summary = (
            f"Score Breakdown for Resume {resume_id} against JD {jd_id} ({jd_row['Title']}):\n"
            f"- Fit Score: {breakdown['baseline_fit_score']:.1f}%\n"
            f"- Experience Gap: {breakdown['experience_gap']:.1f} years (Requires {breakdown['required_experience']:.1f}, Candidate has {breakdown['candidate_experience']:.1f})\n"
            f"- Degree Match: {'Yes' if breakdown['education_fit']['is_match'] else 'No'} (Candidate has {breakdown['education_fit']['candidate_education']}, Job requires {breakdown['education_fit']['required_education']})\n"
            f"- Skill Overlap Ratio: {breakdown['skill_overlap_ratio']*100:.0f}% (Matched {len(breakdown['matched_skills'])} of {len(breakdown['matched_skills']) + len(breakdown['missing_skills'])} skills)\n"
            f"- Explanation: {breakdown['explanation_text']}"
        )
        return summary
    except Exception as e:
        return f"Error computing score breakdown: {str(e)}"

@tool
def get_missing_skills(resume_id: str, jd_id: str) -> str:
    """
    Retrieves the list of skills required by the job description but missing from the resume.
    Use this tool when the user asks about missing skills or what they need to learn.
    """
    try:
        resumes_df, jds_df = _load_data()
        
        # Look up resume and JD
        res_matches = resumes_df[resumes_df['ResumeID'] == resume_id]
        jd_matches = jds_df[jds_df['JobDescriptionID'] == jd_id]
        
        if res_matches.empty:
            return f"Error: Resume ID {resume_id} not found."
        if jd_matches.empty:
            return f"Error: Job Description ID {jd_id} not found."
            
        res_row = res_matches.iloc[0]
        jd_row = jd_matches.iloc[0]
        
        cand_skills = set(s.lower() for s in res_row.get('ExtractedSkills', []))
        req_skills = jd_row.get('RequiredSkills', [])
        
        missing_skills = [s for s in req_skills if s.lower() not in cand_skills]
        matched_skills = [s for s in req_skills if s.lower() in cand_skills]
        
        if not missing_skills:
            return f"The candidate matches all required skills for this role ({', '.join(matched_skills)})."
            
        return (
            f"Missing Skills for Resume {resume_id} against JD {jd_id} ({jd_row['Title']}):\n"
            f"- Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}\n"
            f"- Missing Skills: {', '.join(missing_skills)}"
        )
    except Exception as e:
        return f"Error retrieving missing skills: {str(e)}"

@tool
def get_similar_resumes(resume_id: str) -> str:
    """
    Retrieves similar candidate resumes in the same category from the FAISS database.
    Use this tool when the user asks about peer resumes, benchmarks, or how other candidates compare.
    """
    try:
        resumes_df, _ = _load_data()
        _, _, _, _, faiss_db = _load_models()
        
        res_matches = resumes_df[resumes_df['ResumeID'] == resume_id]
        if res_matches.empty:
            return f"Error: Resume ID {resume_id} not found."
            
        res_row = res_matches.iloc[0]
        
        from src.feedback.generate import retrieve_similar_resumes as get_peers
        peers = get_peers(res_row, faiss_db, resumes_df, k=3)
        
        if not peers:
            return "No similar peer resumes found in the database."
            
        summary_lines = [f"Similar resumes in the same category ({res_row['Category']}):"]
        for p in peers:
            skills = ", ".join(p.get('ExtractedSkills', [])[:5])
            summary_lines.append(
                f"- Candidate {p['ResumeID']} | Experience: {p.get('YearsOfExperience', 0.0):.1f} years | Degree: {p.get('DegreeLevel', 'None')}\n"
                f"  Skills: {skills}"
            )
            
        return "\n".join(summary_lines)
    except Exception as e:
        return f"Error retrieving similar resumes: {str(e)}"

@tool
def rewrite_suggestion(section_text: str) -> str:
    """
    Rewrites a weak resume bullet point or experience section text using resume-writing best practices.
    It removes passive wording (e.g. 'Responsible for') and suggests how to quantify impact.
    Use this tool when the user asks to rewrite, rephrase, or improve a bullet point or section of their resume.
    """
    text_clean = section_text.strip()
    
    # Simple rule-based rewriter to support local CPU/non-LLM execution
    suggestions = []
    
    # Check for passive boilerplate
    passive_phrases = ["responsible for", "assisted in", "helped with", "worked on", "duties included"]
    detected_passive = False
    for p in passive_phrases:
        if p in text_clean.lower():
            detected_passive = True
            break
            
    # Check for metrics
    has_metrics = any(c.isdigit() or '%' in text_clean for c in text_clean)
    
    # Construct a rewritten suggestion
    rewritten = text_clean
    if detected_passive:
        # Suggest replacing passive verbs
        rewritten = rewritten.replace("Responsible for writing", "Engineered")
        rewritten = rewritten.replace("responsible for writing", "engineered")
        rewritten = rewritten.replace("Responsible for", "Spearheaded")
        rewritten = rewritten.replace("responsible for", "spearheaded")
        rewritten = rewritten.replace("Assisted in", "Collaborated on")
        rewritten = rewritten.replace("assisted in", "collaborated on")
        rewritten = rewritten.replace("Worked on", "Implemented")
        rewritten = rewritten.replace("worked on", "implemented")
        suggestions.append("Action Words: Replaced passive phrasing with strong, result-oriented action verbs (e.g. 'Engineered', 'Spearheaded').")
        
    if not has_metrics:
        rewritten += " resulting in a 15% increase in team velocity and 20% latency reduction"
        suggestions.append("Quantifying Impact: Added placeholder business outcome metrics. Be sure to replace these with your actual metrics.")
        
    if not suggestions:
        # Generic enhancement
        rewritten = f"Architected and optimized {text_clean}, enhancing system reliability by 25%"
        suggestions.append("Impact Enhancement: Upgraded bullet point to showcase architectural design and system-level impact.")
        
    output = (
        f"Original text: \"{text_clean}\"\n"
        f"Rewritten Suggestion: \"{rewritten}\"\n"
        f"Applied Best Practices:\n" + "\n".join(f"- {s}" for s in suggestions)
    )
    return output

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: int
    active_resume_id: str
    active_jd_id: str

# System prompt for the ReAct agent
SYSTEM_PROMPT = """You are ResumeIQ, an expert AI career coach.
You answer user follow-up questions about their resume matching and fit score.
You MUST strictly base your answers on the outputs of the tools provided. Do not fabricate scores, skills, or similar resumes from your parametric knowledge.
Always use the get_score_breakdown, get_missing_skills, get_similar_resumes, or rewrite_suggestion tools to retrieve the facts before answering.

The user's active session is initialized with:
- Active Resume ID: {active_resume_id}
- Active Job Description ID: {active_jd_id}

If the user asks questions referencing 'my score', 'my gaps', or 'my resume' without specifying IDs, pass the active IDs above to the tools.
"""

from langchain_core.messages import SystemMessage

def create_agent(llm):
    """Assembles and compiles the ReAct agent graph with memory checkpointer."""
    tools = [get_score_breakdown, get_missing_skills, get_similar_resumes, rewrite_suggestion]
    memory = MemorySaver()
    
    def _prompt_modifier(state: AgentState) -> List[BaseMessage]:
        sys_prompt = SYSTEM_PROMPT.format(
            active_resume_id=state.get("active_resume_id", "REAL_0001"),
            active_jd_id=state.get("active_jd_id", "JD_0001")
        )
        return [SystemMessage(content=sys_prompt)] + list(state.get("messages", []))
        
    agent = create_react_agent(
        llm, 
        tools, 
        state_schema=AgentState,
        prompt=_prompt_modifier,
        checkpointer=memory
    )
    return agent
