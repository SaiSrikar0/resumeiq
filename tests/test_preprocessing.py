import pytest
import pandas as pd
from src.preprocessing.cleaner import clean_whitespace, strip_contact_boilerplate, clean_ocr_text, deduplicate_resumes
from src.preprocessing.extractor import extract_skills, extract_years_experience, extract_degree

def test_clean_whitespace():
    assert clean_whitespace("hello   world \t new\nline") == "hello world new line"
    assert clean_whitespace("") == ""
    assert clean_whitespace(None) == ""

def test_strip_contact_boilerplate():
    text = "jessica claire montgomery street san francisco ca 94105 555 4321000 resumesampleexamplecom professional summary highly skilled"
    cleaned = strip_contact_boilerplate(text)
    # The street address, zip code, phone number and website should be stripped
    assert "jessica claire" not in cleaned
    assert "94105" not in cleaned
    assert "resumesampleexamplecom" not in cleaned
    assert "professional summary highly skilled" in cleaned

def test_clean_ocr_text():
    text = "John Doe | java * developer • 555-123-4567 • john@doe.com"
    cleaned = clean_ocr_text(text)
    assert "|" not in cleaned
    assert "*" not in cleaned
    assert "•" not in cleaned
    assert "555-123-4567" not in cleaned
    assert "john@doe.com" not in cleaned
    assert "java developer" in cleaned

def test_deduplicate_resumes():
    df = pd.DataFrame({
        "ResumeID": ["R1", "R2", "R3"],
        "Text": [
            "This is a unique resume for python developer with django and flask.",
            "This is a unique resume for python developer with django and flask.", # Duplicate of R1
            "Totally different resume about network security, firewalls and routing."
        ]
    })
    df_dedup = deduplicate_resumes(df, text_col="Text", threshold=0.9)
    assert len(df_dedup) == 2
    assert "R1" in df_dedup["ResumeID"].values
    assert "R3" in df_dedup["ResumeID"].values
    assert "R2" not in df_dedup["ResumeID"].values

def test_extract_skills():
    text = "Experienced with Python, Django, React and some SQL databases."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "Django" in skills
    assert "React" in skills
    assert "SQL" in skills
    assert "Java" not in skills

def test_extract_years_experience():
    # Test direct mention
    t1 = "Senior engineer with 8+ years of experience in system design."
    assert extract_years_experience(t1) == 8.0
    
    # Test OCR-style run-together dates
    t2 = "Worked as developer 2014 2018 at Intel and 102011 112013 at Microsoft."
    # 2014-2018 is 4 years. 10/2011 - 11/2013 is 2.1 years. Total is 6.1 years.
    assert extract_years_experience(t2) == 6.1

def test_extract_degree():
    assert extract_degree("Bachelor of Science in CS (B.Sc. or BTech)") == "Bachelor's"
    assert extract_degree("Master of Business Administration (MBA)") == "Master's"
    assert extract_degree("Ph.D. in Machine Learning and NLP") == "PhD"
    assert extract_degree("High School Diploma") == "None"
