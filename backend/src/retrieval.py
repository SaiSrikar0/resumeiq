import os
import json
import pickle
import numpy as np
import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from src.models import load_model, get_embeddings_batched

# Curated set of 30 resume-writing best-practice snippets
BEST_PRACTICES = [
    # Action Verbs & Active Language
    "Use strong action verbs like 'Engineered', 'Optimized', 'Designed', or 'Architected' at the beginning of each bullet point to project initiative.",
    "Avoid passive phrases like 'Responsible for' or 'Assisted in'; instead, use direct verbs like 'Executed', 'Spearheaded', or 'Implemented'.",
    "Describe your projects using the CAR (Context, Action, Result) structure to highlight the direct impact of your contributions.",
    "Ensure each job description entry starts with a past-tense action verb (or present-tense for your current role) to maintain active language.",
    
    # Quantifying Impact
    "Quantify your achievements by including specific percentages, dollar values, or counts (e.g., 'reduced page load time by 30%', 'managed a team of 4').",
    "Focus on business outcome metrics (e.g., revenue generated, database latency reduction, server costs saved) rather than just listing tasks.",
    "Provide scale context for your experience: state database sizes, number of concurrent users, or infrastructure size to demonstrate capability.",
    "Define the scope of the project you worked on, mentioning target client base size or system throughput to ground your impact.",
    
    # Skills Presentation
    "Group technical skills into clear subcategories (e.g., Languages, Frameworks, Cloud/DevOps, Databases) to make the skills section highly scannable.",
    "Only include technical skills that you can explain and support in an interview; do not list technologies you only have passing familiarity with.",
    "Align the terms and spellings in your skills section exactly with the job description (e.g., write 'Spring Boot' instead of 'Springboot') to pass filters.",
    "Incorporate your skills naturally within your experience bullet points to demonstrate how you applied them in practice.",
    
    # Experience Gaps & Length
    "To address an experience gap in years, highlight freelance work, personal open-source contributions, or specialized technical courses completed during that time.",
    "Focus on depth of experience rather than tenure: show how you quickly scaled up on technologies to offset minor experience gaps.",
    "If your experience is slightly short of the requirements, emphasize leadership roles, mentorship, and system design contributions you made.",
    "For roles requiring more years of experience, emphasize your architectural ownership, mentorship of junior developers, and long-term project planning.",
    
    # Education Matching
    "If you do not hold the required formal degree, place certifications (like AWS, CKA, or Scrum Master) and completed bootcamps in a prominent Education/Certifications section.",
    "Highlight relevant college coursework, degree-level online specializations, or thesis work to demonstrate equivalent theoretical knowledge.",
    "If you have significant professional experience, place the Education section at the bottom of your resume to prioritize your work history.",
    
    # Tech-Stack Specific Tips (DevOps)
    "For DevOps roles, highlight experience with Infrastructure as Code (IaC) tools like Terraform, and container orchestration platforms like Kubernetes.",
    "Emphasize your CI/CD pipeline automation achievements, detailing how you reduced deployment time or improved build reliability.",
    
    # Tech-Stack Specific Tips (Python/Data Science)
    "For Data Science and ML roles, explicitly describe the models you trained, the features engineered, and the evaluation metrics achieved (e.g., accuracy, AUC).",
    "Mention the data manipulation libraries (Pandas, NumPy) and machine learning libraries (Scikit-learn, PyTorch, TensorFlow) you utilized.",
    
    # Tech-Stack Specific Tips (Backend/Java)
    "For Java/Backend roles, explain your microservices design pattern experience, database tuning, and API design principles.",
    "Detail how you managed transaction consistency, caching (e.g. Redis), or message queues (e.g. Kafka) in backend systems.",
    
    # Tech-Stack Specific Tips (Frontend/React)
    "For Frontend roles, highlight state management tools (Redux, Context API), responsive CSS frameworks (Tailwind, Bootstrap), and web performance tuning.",
    "Describe your experience with components modularity, reusable hooks, and testing frameworks like Jest or Cypress.",
    
    # General Formatting
    "Limit your resume to 1-2 pages: keep descriptions concise and focused on high-impact projects.",
    "Avoid formatting anomalies like nested tables, complex sidebars, or progress bars for skills, which can cause parser failures.",
    "Proofread carefully to eliminate spelling, grammatical, or punctuation errors, as they convey a lack of attention to detail."
]

class DistilBertEmbeddings(Embeddings):
    """LangChain custom embeddings wrapper around DistilBERT."""
    def __init__(self, model, tokenizer, pooling='mean'):
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        
    def embed_documents(self, texts):
        return get_embeddings_batched(texts, self.model, self.tokenizer, pooling=self.pooling).tolist()
        
    def embed_query(self, text):
        return get_embeddings_batched([text], self.model, self.tokenizer, pooling=self.pooling)[0].tolist()

def build_and_save_index():
    print("Building FAISS Vector Store...")
    tokenizer, model = load_model()
    
    with open('artifacts/embedding_metadata.json', 'r') as f:
        meta = json.load(f)
    best_pooling = meta['best_pooling']
    
    custom_embeddings = DistilBertEmbeddings(model, tokenizer, pooling=best_pooling)
    
    resumes_df = pd.read_parquet('data/processed/processed_resumes.parquet')
    resume_embs = np.load('artifacts/resume_embeddings.npy')
    
    with open('artifacts/resume_ids.json', 'r') as f:
        resume_ids = json.load(f)
        
    text_embeddings = []
    metadatas = []
    
    print("Adding resumes to text embeddings list...")
    for i, row in resumes_df.iterrows():
        res_id = row['ResumeID']
        res_idx = resume_ids.index(res_id)
        emb = resume_embs[res_idx].tolist()
        
        text_content = f"Resume ID: {res_id}\nCategory: {row['Category']}\nText: {row['Text']}"
        text_embeddings.append((text_content, emb))
        
        metadatas.append({
            'type': 'resume',
            'id': res_id,
            'category': row['Category'],
            'skills': row.get('ExtractedSkills', []),
            'experience': float(row.get('YearsOfExperience', 0.0)),
            'degree': row.get('DegreeLevel', 'None')
        })
        
    print("Embedding best practice snippets...")
    bp_embs = custom_embeddings.embed_documents(BEST_PRACTICES)
    
    for i, bp in enumerate(BEST_PRACTICES):
        bp_id = f"BP_{i:04d}"
        text_embeddings.append((bp, bp_embs[i]))
        metadatas.append({
            'type': 'best_practice',
            'id': bp_id,
            'category': 'general'
        })
        
    print("Creating FAISS index...")
    db = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=custom_embeddings,
        metadatas=metadatas
    )
    
    os.makedirs('artifacts', exist_ok=True)
    db.save_local('artifacts/faiss_index')
    print("FAISS index saved to artifacts/faiss_index")
    return db

def load_index(model=None, tokenizer=None):
    """Loads the local FAISS index from disk."""
    if model is None or tokenizer is None:
        tokenizer, model = load_model()
        
    with open('artifacts/embedding_metadata.json', 'r') as f:
        meta = json.load(f)
    best_pooling = meta['best_pooling']
    
    custom_embeddings = DistilBertEmbeddings(model, tokenizer, pooling=best_pooling)
    db = FAISS.load_local(
        'artifacts/faiss_index',
        custom_embeddings,
        allow_dangerous_deserialization=True
    )
    return db

def main():
    build_and_save_index()

if __name__ == '__main__':
    main()
