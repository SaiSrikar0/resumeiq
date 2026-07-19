import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

# Helper function to ensure app is initialized (runs startup events)
@pytest.fixture(scope="module", autouse=True)
def init_app():
    with TestClient(app) as c:
        yield c

def test_list_jobs():
    response = client.get("/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "synthetic" in data["data_source_label"].lower()
    assert len(data["jobs"]) == 40
    assert data["jobs"][0]["type"] == "synthetic"
    assert "jd_id" in data["jobs"][0]

def test_upload_resume_text_happy():
    response = client.post("/resumes", data={"text": "This is a resume for a Java Developer with 5 years experience in Spring Boot and SQL.", "category": "Java Developer"})
    assert response.status_code == 200
    data = response.json()
    assert "resume_id" in data
    assert data["status"] == "success"
    assert "skills" in data["extracted_features"]
    assert "java" in [s.lower() for s in data["extracted_features"]["skills"]]

def test_upload_resume_file_happy():
    import io
    file_data = io.BytesIO(b"Resume content text details for an expert Data Scientist with Python, PyTorch, and SQL experience.")
    response = client.post(
        "/resumes",
        data={"category": "Data Science"},
        files={"file": ("resume.txt", file_data, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "resume_id" in data
    assert "python" in [s.lower() for s in data["extracted_features"]["skills"]]

def test_upload_resume_validation_fail():
    # Empty payload
    response = client.post("/resumes", data={})
    assert response.status_code == 400
    # Short text payload
    response = client.post("/resumes", data={"text": "short"})
    assert response.status_code == 400

def test_score_endpoints_happy():
    # 1. First upload a resume to get a custom ID
    upload_resp = client.post("/resumes", data={"text": "Expert Java Developer resume with Spring Boot and SQL. 5 years experience.", "category": "Java Developer"})
    res_id = upload_resp.json()["resume_id"]
    
    # 2. Score against synthetic JD
    score_resp = client.post("/score", json={"resume_id": res_id, "jd_id": "JD_0001"})
    assert score_resp.status_code == 200
    score_data = score_resp.json()
    assert score_data["resume_id"] == res_id
    assert "score_breakdown" in score_data
    assert "baseline_fit_score" in score_data["score_breakdown"]
    
    # 3. Score against ad-hoc custom JD text
    score_custom_resp = client.post("/score", json={
        "resume_id": res_id, 
        "jd_text": "We need a Software Engineer with Python and Docker experience. Requires 3 years experience and a Bachelors degree."
    })
    assert score_custom_resp.status_code == 200
    score_custom_data = score_custom_resp.json()
    assert "CUSTOM_" in score_custom_data["jd_id"]
    assert "python" in [s.lower() for s in score_custom_data["jd_details"]["required_skills"]]

def test_score_validation_fail():
    # Non-existent resume_id
    response = client.post("/score", json={"resume_id": "INVALID_ID", "jd_id": "JD_0001"})
    assert response.status_code == 404
    # Non-existent jd_id
    response = client.post("/score", json={"resume_id": "REAL_0001", "jd_id": "INVALID_JD"})
    assert response.status_code == 404

def test_feedback_endpoint_happy():
    response = client.post("/feedback", json={"resume_id": "REAL_0001", "jd_id": "JD_0001"})
    assert response.status_code == 200
    data = response.json()
    assert "feedback_report" in data
    assert "# ResumeIQ Feedback Report" in data["feedback_report"]

def test_chat_endpoint_happy():
    # Ask about score breakdown
    response = client.post("/chat", json={
        "resume_id": "REAL_0001",
        "jd_id": "JD_0001",
        "session_id": "test_session_123",
        "message": "Why did I get this score? Can you break it down for me?"
    })
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["tool_calls"]) > 0
    assert data["tool_calls"][0]["name"] == "get_score_breakdown"

    # Ask about missing skills
    response_skills = client.post("/chat", json={
        "resume_id": "REAL_0001",
        "jd_id": "JD_0001",
        "session_id": "test_session_123",
        "message": "What missing skills should I learn?"
    })
    assert response_skills.status_code == 200
    data_skills = response_skills.json()
    assert "get_missing_skills" in [tc["name"] for tc in data_skills["tool_calls"]]
