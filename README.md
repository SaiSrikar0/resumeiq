# ResumeIQ

ResumeIQ is an end-to-end, locally-runnable AI system that evaluates candidate resumes against job requirements, produces personalized, explainable feedback, and provides a conversational interface for candidates to discuss their results.

## Architecture

```mermaid
graph TD
    A[Candidate Resume] --> E[Preprocessing Pipeline]
    B[Job Description] --> E
    E --> F[Feature & Skill Extraction]
    F --> G[Processed Resumes Parquet]
    
    G --> H[Classical ML Baseline: XGBoost/LightGBM]
    G --> I[Deep Learning Matcher: BERT Embeddings + Attention]
    
    H --> J[Explainability Layer: Feature Importance]
    I --> K[Explainability Layer: Attention weights]
    
    J & K --> L[RAG Feedback Pipeline]
    M[Best Writing Practices Snippets] --> N[FAISS Vector Index]
    G --> N
    N --> L
    
    L --> O[Conversational Interface: LangGraph Agent]
    O --> P[FastAPI Backend / React Dashboard]
```

## Project Structure

```text
resumeiq/
  README.md
  requirements.txt
  .gitignore
  data/
    raw/                 # Raw resumes_dataset.jsonl seed
    processed/           # Processed resumes parquet + job descriptions JSONL
  artifacts/             # Unified models, vectorizer, & FAISS index storage
  src/
    preprocessing.py     # OCR cleaning + entity extraction library
    prepare_data.py      # Standalone data prep & JDs synthesis script (run once)
    models.py            # ML baseline models + DistilBERT encoders
    retrieval.py         # FAISS retrieval + RAG practices indexer
    feedback.py          # Fit scoring breakdown & feedback generator
    agent.py             # LangGraph ReAct conversational agent
    api.py               # FastAPI backend
    run_pipeline.py      # CLI evaluation pipeline runner
  frontend/              # React frontend using Tailwind CSS v4
  tests/                 # PyTest suite
  docker/                # Docker files & docker-compose configurations
```

## Setup & Running

### Prerequisites
- Python 3.13.5
- Node.js (for frontend)
- Ollama (for local LLM feedback generation, e.g. Llama 3.1 or Mistral)

### Installation
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running EDA (Phase 1)
To run the preprocessing, cleaning, and EDA metrics printout:
```bash
python src/preprocessing/run_eda.py
```

### Running Tests
To run the automated tests covering preprocessing and extraction:
```bash
pytest tests/
```

## Phase 1 EDA Summary
The corrected dataset contains 3,124 unique resumes after removing 376 near-duplicate resumes from the original 3,500 entries.
- **Completeness**: All raw fields (ResumeID, Category, Name, Email, Phone, Location, Summary, Skills, Experience, Education, Text, Source) are 100% complete.
- **Top 5 Categories**:
  1. Data Science: 170 (5.44%)
  2. Java Developer: 168 (5.38%)
  3. Python Developer: 161 (5.15%)
  4. SQL Developer: 149 (4.77%)
  5. DevOps: 140 (4.48%)
- **Cleaned Text Length Distribution**:
  - Min Length: 202 characters
  - Max Length: 55,681 characters
  - Mean Length: 3,126 characters
  - Median Length: 1,845 characters

## How Scoring Works
The ResumeIQ fit score is a structured, rebalanced composite score designed to weigh candidate qualifications fairly and prevent keyword stuffing. It is **not** a single end-to-end learned prediction.

The score is computed as:
$$\text{Fit Score} = 0.40 \cdot \text{Skill Overlap} + 0.30 \cdot \text{Experience Match} + 0.15 \cdot \text{Degree Match} + 0.15 \cdot \text{Model Semantic Score}$$

### Sub-Score Explanations:
1. **Skill Overlap (40% weight)**: Measures the direct technical keyword overlap between the candidate's extracted skills and the required skills list ($\text{matched\_skills} / \text{required\_skills}$).
2. **Experience Match (30% weight)**: Evaluates the candidate's years of experience against the required experience. If the candidate meets or exceeds the requirement, they receive a score of $1.0$. If there is an experience gap, the score is reduced proportionally ($\max(0.0, 1.0 - \frac{\text{gap}}{\text{required\_experience}}$).
3. **Degree Match (15% weight)**: A binary check ($1.0$ or $0.0$) evaluating whether the candidate's highest degree level (None, Bachelor's, Master's, PhD) meets or exceeds the required education level.
4. **Model Semantic Score (15% weight)**: The probability output of the trained model (incorporating TF-IDF cosine similarity), representing overall vocabulary similarity and semantic alignment.

## Known Limitations & Weak Labels
1. **Synthetic Seed JDs**: Since the raw resumes dataset did not contain target job descriptions, a companion set of 40 job descriptions was programmatically synthesized and saved to `data/job_descriptions.jsonl`.
2. **Category as Weak Labels**: The candidate category is used as a weak label proxy for "best-fit role" matching, meaning the models are trained to predict the category match rather than validated hiring success.
3. **No spaCy/SHAP**: In compliance with technical constraints, no external NER libraries (like spaCy) or SHAP explainers are used. Features are extracted using regex and custom dictionary mappings, and explainability is built using model feature importances and BERT multi-head attention scores.
4. **Zero-Experience Resumes**: Approximately 39% of the resumes in the corpus do not state numeric years of experience or contain unfilled template placeholders (e.g. "bringing number years experience"), resulting in an extracted experience of 0.0 years. This is a known dataset limitation.
5. **LoRA fine-tuning**: Not attempted — Bypassed to maintain containerized runtime stability and avoid high CPU overhead during contrastive training in the deployment environment.

## Docker Containerization

ResumeIQ can be run inside a fully self-contained Docker Compose network.

### Steps to Run:
1. Ensure Docker Desktop is installed and running on your system.
2. Build and start the services:
   ```bash
   docker-compose up --build
   ```
3. Once running, open the applications:
   - **Interactive Frontend Dashboard**: `http://localhost:3000`
   - **FastAPI Backend Swagger Docs**: `http://localhost:8000/docs`

### Configuring Retrained Models:
If you retrain the baseline LightGBM/XGBoost models or regenerate the FAISS index, you can point the docker container at your updated artifacts without rebuilding the image by using volume mounts in `docker-compose.yml`:
```yaml
  backend:
    volumes:
      - ./artifacts:/app/artifacts
```

## Sample Feedback Report

Here is a sample feedback report generated by ResumeIQ for candidate `REAL_0001` (Java Developer) matched against `JD_0001` (Senior Java Developer):

<details>
<summary>Click to view Sample Feedback Report</summary>

# ResumeIQ Feedback Report
**Target Role:** Senior Java Developer (Java Developer)
**Candidate ID:** REAL_0001 | **Fit Score:** 55.4%

## Score & Explainability Summary
ResumeIQ calculated a fit score of 55.4% for this role, driven primarily by a skill overlap of 17% and semantic text similarity. The candidate matches core skills such as Java. However, key required skills like CI/CD, Docker, Git, and 2 other(s) are missing. The candidate's 12.0 years of experience successfully meets the required 5.0 years. The candidate's education (Bachelor's) meets or exceeds the required Bachelor's. Additionally, the neural model's attention highlights that terminology surrounding 'implemented, including, designs' in the resume was highly influential during matching.

## Actionable Suggestions (Grounded in Gaps)
### Technical Skills Alignment
*Gaps Addressed: Missing Skills (CI/CD, Docker, Git, SQL, Spring Boot)*
**[Addresses Gap: Missing Skills]** Your resume is currently missing key skills listed in the job description: **CI/CD, Docker, Git, SQL, Spring Boot**.
- Group technical skills into clear subcategories (e.g., Languages, Frameworks, Cloud/DevOps, Databases) to make the skills section highly scannable.

### General Formatting & Action Words
**[Addresses Gap: General Scannability]** To maximize parsing rates and highlight your strengths:
- Mention the data manipulation libraries (Pandas, NumPy) and machine learning libraries (Scikit-learn, PyTorch, TensorFlow) you utilized.

## Peer References
Here are details from similar candidates in your category who scored higher against this job description. Use their phrasing and skill highlights as references:
- **Candidate REAL_0108** (Fit Score: **94.5%** | Experience: **0.0 years**)
  - *Skills Emphasized:* Java, JavaScript, SQL, HTML, CSS
  - *Summary Excerpt:* "junior java developer robert smith phone 123 456 78 99 email infoqwikresumecom website wwwqwikresumecom alabama objective 2 plus years exper..."
- **Candidate REAL_0091** (Fit Score: **90.7%** | Experience: **5.0 years**)
  - *Skills Emphasized:* Java, SQL, Spring Boot, Oracle, MongoDB
  - *Summary Excerpt:* "certified java developer 5 years experience developing apps various industries seeking help techvantage ..."
- **Candidate REAL_0066** (Fit Score: **83.0%** | Experience: **5.0 years**)
  - *Skills Emphasized:* Java, HTML, CSS, Spring Boot, Git
  - *Summary Excerpt:* "chicago illinois us linkedincomresumekraft summary highly skilled dedicated java developer 5 years experience designing developing applicatio..."

</details>
