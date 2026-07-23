# ResumeIQ

ResumeIQ is a local resume-to-job-description matching app. It scores a resume, explains the result, and shows the feedback in a web UI.

## What it does
- Scores resumes against a job description
- Explains the score with matched and missing skills
- Generates feedback and a simple chat-style response layer
- Provides a React dashboard backed by a FastAPI API

## Requirements
- Python 3.13.5
- Node.js
- Ollama is optional and only needed for local LLM-style text generation

## Setup
Install backend dependencies:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

Prepare the data once:

```bash
cd backend
python -m src.cli prepare-data
```

## Run the app
Start the backend:

```bash
cd backend
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Start the frontend in a second terminal:

```bash
cd frontend
npm run dev
```

Open the Vite URL printed in the terminal, usually http://localhost:5173.

## Useful commands
Run the full pipeline from the backend folder:

```bash
python -m src.cli run-pipeline
```

Run tests:

```bash
cd backend
python -m pytest tests/ -v
```

## Project layout
```text
backend/
  src/
    api.py
    cli.py
    feedback.py
    models.py
    preprocessing.py
    retrieval.py
  tests/
frontend/
  src/
```

## Notes
- The project uses synthetic job descriptions for demos and testing.
- If Ollama is not available, the feedback/chat experience falls back to a simpler local mode.