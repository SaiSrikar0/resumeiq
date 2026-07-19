import os
import json
import uuid
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import clean_ocr_text, extract_structured_fields, extract_skills, extract_years_experience, extract_degree
from src.feedback import load_baseline, compute_explainability_breakdown, generate_feedback_report
from src.retrieval import load_index
from src.models import load_model
from src.agent import create_agent

app = FastAPI(
    title="ResumeIQ API",
    description="Backend API for ResumeIQ resume match screening, feedback generation, and career agent chat.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.custom_resumes = {}

@app.on_event("startup")
def startup_event():
    print("Loading ResumeIQ datasets and model artifacts...")
    app.state.resumes_df = pd.read_parquet('data/processed/processed_resumes.parquet')
    app.state.jds_df = pd.read_json('data/processed/job_descriptions.jsonl', lines=True)
    
    app.state.baseline_clf, app.state.baseline_type, app.state.importances = load_baseline()
    with open('artifacts/tfidf_vectorizer.pkl', 'rb') as f:
        app.state.tfidf = pickle.load(f)
        
    app.state.tokenizer, app.state.emb_model = load_model()
    app.state.faiss_db = load_index(app.state.emb_model, app.state.tokenizer)
    print("Startup complete. All models loaded.")

class ResumeUploadText(BaseModel):
    text: str = Field(..., min_length=10, description="Raw text of the resume")
    category: Optional[str] = Field("Unknown", description="Job category of the candidate")

class ScoreRequest(BaseModel):
    resume_id: str = Field(..., description="ID of the resume (e.g. REAL_0001 or custom generated ID)")
    jd_id: Optional[str] = Field(None, description="ID of the job description from GET /jobs")
    jd_text: Optional[str] = Field(None, description="Ad-hoc job description text")

class FeedbackRequest(BaseModel):
    resume_id: str
    jd_id: Optional[str] = None
    jd_text: Optional[str] = None

class ChatRequest(BaseModel):
    resume_id: str
    jd_id: str
    session_id: str
    message: str

def get_resume_by_id(resume_id: str) -> Dict[str, Any]:
    if resume_id in app.state.custom_resumes:
        return app.state.custom_resumes[resume_id]
        
    matches = app.state.resumes_df[app.state.resumes_df['ResumeID'] == resume_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Resume ID '{resume_id}' not found.")
    return matches.iloc[0].to_dict()

def get_job_description(jd_id: Optional[str], jd_text: Optional[str]) -> Dict[str, Any]:
    if not jd_id and not jd_text:
        raise HTTPException(status_code=400, detail="Must provide either jd_id or jd_text.")
        
    if jd_id:
        matches = app.state.jds_df[app.state.jds_df['JobDescriptionID'] == jd_id]
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"Job Description ID '{jd_id}' not found.")
        return matches.iloc[0].to_dict()
        
    cleaned_jd = clean_ocr_text(jd_text)
    if not cleaned_jd:
        raise HTTPException(status_code=400, detail="Provided jd_text is empty or invalid.")
        
    skills = extract_skills(cleaned_jd)
    exp = extract_years_experience(cleaned_jd)
    deg = extract_degree(cleaned_jd)
    
    return {
        "JobDescriptionID": f"CUSTOM_{uuid.uuid4().hex[:6].upper()}",
        "Category": "Custom",
        "Title": "Custom Job Role",
        "RequiredSkills": skills,
        "RequiredExperience": exp,
        "RequiredEducation": deg,
        "Description": cleaned_jd
    }

@app.post("/resumes", response_model=Dict[str, Any])
async def upload_resume(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    category: str = Form("Unknown")
):
    resume_content = ""
    if file:
        content_bytes = await file.read()
        try:
            resume_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Uploaded file must be UTF-8 encoded text.")
    elif text:
        resume_content = text
    else:
        raise HTTPException(status_code=400, detail="Must provide either text body or a file.")
        
    cleaned_text = clean_ocr_text(resume_content)
    if len(cleaned_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Resume text is too short or empty after cleaning.")
        
    temp_row = {
        "Summary": cleaned_text,
        "Experience": "",
        "Education": "",
        "Text": cleaned_text
    }
    extracted = extract_structured_fields(temp_row)
    
    if extracted["DegreeLevel"] == "None":
        extracted["DegreeLevel"] = extract_degree(cleaned_text)
        
    resume_id = f"TEMP_{uuid.uuid4().hex[:6].upper()}"
    
    resume_record = {
        "ResumeID": resume_id,
        "Category": category,
        "Text": cleaned_text,
        "ExtractedSkills": extracted["ExtractedSkills"],
        "YearsOfExperience": extracted["YearsOfExperience"],
        "DegreeLevel": extracted["DegreeLevel"]
    }
    
    app.state.custom_resumes[resume_id] = resume_record
    return {
        "status": "success",
        "resume_id": resume_id,
        "extracted_features": {
            "skills": resume_record["ExtractedSkills"],
            "experience": resume_record["YearsOfExperience"],
            "degree": resume_record["DegreeLevel"]
        }
    }

@app.post("/score")
def score_resume(req: ScoreRequest):
    res_row = get_resume_by_id(req.resume_id)
    jd_row = get_job_description(req.jd_id, req.jd_text)
    
    breakdown = compute_explainability_breakdown(
        res_row, 
        jd_row, 
        app.state.baseline_clf, 
        app.state.baseline_type, 
        app.state.importances, 
        app.state.tfidf, 
        app.state.emb_model, 
        app.state.tokenizer
    )
    
    return {
        "resume_id": req.resume_id,
        "jd_id": jd_row["JobDescriptionID"],
        "jd_details": {
            "title": jd_row["Title"],
            "category": jd_row["Category"],
            "required_skills": jd_row["RequiredSkills"],
            "required_experience": jd_row["RequiredExperience"],
            "required_education": jd_row["RequiredEducation"]
        },
        "score_breakdown": breakdown
    }

@app.post("/feedback")
def get_feedback(req: FeedbackRequest):
    res_row = get_resume_by_id(req.resume_id)
    jd_row = get_job_description(req.jd_id, req.jd_text)
    
    breakdown = compute_explainability_breakdown(
        res_row, 
        jd_row, 
        app.state.baseline_clf, 
        app.state.baseline_type, 
        app.state.importances, 
        app.state.tfidf, 
        app.state.emb_model, 
        app.state.tokenizer
    )
    
    feedback_report = generate_feedback_report(
        res_row, 
        jd_row, 
        breakdown, 
        app.state.faiss_db, 
        app.state.resumes_df, 
        app.state.baseline_clf, 
        app.state.baseline_type, 
        app.state.tfidf, 
        use_llm=False
    )
    
    return {
        "resume_id": req.resume_id,
        "jd_id": jd_row["JobDescriptionID"],
        "feedback_report": feedback_report
    }

@app.post("/chat")
def run_chat(req: ChatRequest):
    res_row = get_resume_by_id(req.resume_id)
    jd_row = get_job_description(req.jd_id, None)
    
    user_msg = req.message.lower()
    tool_calls_executed = []
    
    from src.agent import get_score_breakdown, get_missing_skills, get_similar_resumes, rewrite_suggestion
    
    response_text = ""
    
    if "score" in user_msg or "breakdown" in user_msg or "fit" in user_msg:
        tool_calls_executed.append({"name": "get_score_breakdown", "args": {"resume_id": req.resume_id, "jd_id": req.jd_id}})
        resp = get_score_breakdown.invoke({"resume_id": req.resume_id, "jd_id": req.jd_id})
        response_text = f"Here is your matching score breakdown:\n\n{resp}"
    elif "missing" in user_msg or "skills" in user_msg or "gap" in user_msg:
        tool_calls_executed.append({"name": "get_missing_skills", "args": {"resume_id": req.resume_id, "jd_id": req.jd_id}})
        resp = get_missing_skills.invoke({"resume_id": req.resume_id, "jd_id": req.jd_id})
        response_text = f"Here are the key missing skills required for this position:\n\n{resp}"
    elif "similar" in user_msg or "peer" in user_msg or "benchmark" in user_msg:
        tool_calls_executed.append({"name": "get_similar_resumes", "args": {"resume_id": req.resume_id, "jd_id": req.jd_id}})
        resp = get_similar_resumes.invoke({"resume_id": req.resume_id, "jd_id": req.jd_id})
        response_text = f"Here are some similar candidate benchmarks in our database:\n\n{resp}"
    elif "rewrite" in user_msg or "improve" in user_msg or "phrase" in user_msg:
        tool_calls_executed.append({"name": "rewrite_suggestion", "args": {"section_text": req.message}})
        resp = rewrite_suggestion.invoke({"section_text": req.message})
        response_text = f"Here is my suggested optimization for that section:\n\n{resp}"
    else:
        response_text = (
            "Hello! I am your ResumeIQ career coach. I can help explain your match results. "
            "Try asking questions like:\n"
            "- 'Can you break down my fit score?'\n"
            "- 'Which required skills am I missing?'\n"
            "- 'Can you show me similar benchmark profiles?'\n"
            "- 'How can I rewrite this point: \"Responsible for writing code\"?'"
        )
        
    return {
        "session_id": req.session_id,
        "response": response_text,
        "tool_calls": tool_calls_executed
    }

@app.get("/jobs")
def list_jobs():
    jds = []
    for _, row in app.state.jds_df.iterrows():
        jds.append({
            "jd_id": row["JobDescriptionID"],
            "title": row["Title"],
            "category": row["Category"],
            "required_experience": row["RequiredExperience"],
            "required_education": row["RequiredEducation"],
            "required_skills": row["RequiredSkills"],
            "description": row["Description"],
            "type": "synthetic"
        })
    return {
        "status": "success",
        "data_source_label": "Disclaimer: All job descriptions in this list are programmatically generated and synthetic.",
        "jobs": jds
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
