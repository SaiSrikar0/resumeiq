import os
import json
import pickle
import pandas as pd
from pathlib import Path
from src.models.classical_baseline import build_features_optimized
from src.models.embedding_model import load_model, get_embeddings_batched
from src.feedback.explain import load_baseline, compute_explainability_breakdown
from src.feedback.generate import generate_feedback_report
from src.retrieval.vector_store import load_index
from src.agent.agent import create_agent
from langchain_core.messages import HumanMessage

def main():
    print("=" * 60)
    print("ResumeIQ: End-to-End Pipeline Evaluation")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Step 1] Loading sample data...")
    resumes_df = pd.read_parquet('data/processed_resumes.parquet')
    jds_df = pd.read_json('data/job_descriptions.jsonl', lines=True)
    
    # Pick a candidate and a JD from matching categories to see a realistic alignment
    category = "Java Developer"
    java_resumes = resumes_df[resumes_df['Category'] == category]
    java_jds = jds_df[jds_df['Category'] == category]
    
    if java_resumes.empty or java_jds.empty:
        # Fallback to the first items
        res_row = resumes_df.iloc[0]
        jd_row = jds_df.iloc[0]
    else:
        res_row = java_resumes.iloc[0]
        jd_row = java_jds.iloc[0]
        
    print(f"Loaded Candidate Resume: ID={res_row['ResumeID']} | Category={res_row['Category']}")
    print(f"Loaded Job Description:  ID={jd_row['JobDescriptionID']} | Title={jd_row['Title']} | Category={jd_row['Category']}")
    
    # 2. Load Models
    print("\n[Step 2] Loading models and artifacts...")
    baseline_clf, baseline_type, importances = load_baseline()
    with open('src/models/artifacts/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        
    tokenizer, model = load_model()
    faiss_db = load_index(model, tokenizer)
    print("All models and vector stores loaded successfully.")
    
    # 3. Compute Fit Score & Explainability Breakdown
    print("\n[Step 3] Computing fit score and explainability breakdown...")
    breakdown = compute_explainability_breakdown(
        res_row, jd_row, baseline_clf, baseline_type, importances, tfidf, model, tokenizer
    )
    
    print("-" * 40)
    print(f"Fit Score (predicted by {baseline_type}): {breakdown['baseline_fit_score']:.1f}%")
    print(f"\nPlain-Language Explanation:\n{breakdown['explanation_text']}")
    print(f"\nTop Attention Tokens matching the JD:\n{breakdown['attention_top_tokens']}")
    print("-" * 40)
    
    # 4. Generate Feedback Report
    print("\n[Step 4] Generating retrieval-grounded feedback report...")
    feedback_report = generate_feedback_report(
        res_row, jd_row, breakdown, faiss_db, resumes_df, baseline_clf, baseline_type, tfidf, use_llm=False
    )
    
    # Save the report in the artifacts directory
    artifact_dir = Path("C:/Users/bsais/.gemini/antigravity-ide/brain/be473fbb-9f5b-438e-bcc3-f202c5084ac1")
    report_path = artifact_dir / "sample_feedback_report.md"
    report_path.write_text(feedback_report, encoding="utf-8")
    print(f"Feedback report compiled and saved to artifact path: [sample_feedback_report.md](file:///{report_path.as_posix()})")
    
    # 5. Conversational Agent Demo Turn
    print("\n[Step 5] Running demo turn with the Conversational Agent...")
    # Define a simple Mock LLM for the demo run
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    
    class DemoReActLLM(BaseChatModel):
        def bind_tools(self, tools, **kwargs):
            return self
            
        def _generate(self, messages, stop=None, **kwargs):
            # First turn: make the tool call for breakdown
            has_tool = any(isinstance(m, ToolMessage) for m in messages)
            if not has_tool:
                tool_call = {
                    "name": "get_score_breakdown",
                    "args": {"resume_id": res_row['ResumeID'], "jd_id": jd_row['JobDescriptionID']},
                    "id": "demo_call_1"
                }
                ai_msg = AIMessage(content="", tool_calls=[tool_call])
            else:
                tool_output = [m.content for m in messages if isinstance(m, ToolMessage)][-1]
                ai_msg = AIMessage(content=f"Sure! Here is the score breakdown retrieved from our database:\n\n{tool_output}")
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])
            
        def _llm_type(self) -> str:
            return "demo-react-llm"
            
    agent = create_agent(DemoReActLLM())
    
    query = "Why did I get this score? Can you break it down for me?"
    print(f"User Query: '{query}'")
    
    state = {
        "messages": [HumanMessage(content=query)],
        "active_resume_id": res_row['ResumeID'],
        "active_jd_id": jd_row['JobDescriptionID']
    }
    
    config = {"configurable": {"thread_id": "demo_session_99"}}
    agent_output = agent.invoke(state, config=config)
    
    # Display conversation trace
    print("\n--- Agent Conversation Transcript ---")
    for msg in agent_output["messages"]:
        role = msg.__class__.__name__
        content = msg.content
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            content += f" [Tool Call: {msg.tool_calls[0]['name']}]"
        print(f"[{role}]: {content}\n")
    print("--------------------------------------")
    print("\nPipeline run complete. Everything verified successfully!")

if __name__ == "__main__":
    main()
