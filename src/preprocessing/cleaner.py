import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_whitespace(text: str) -> str:
    """Normalize whitespace by replacing multiple spaces, tabs, and newlines with a single space."""
    if not isinstance(text, str):
        return ""
    # Replace multiple whitespaces, tabs, newlines with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def strip_contact_boilerplate(text: str) -> str:
    """
    Strips boilerplate contact header text that leaks into Summary or Experience.
    This includes emails, phone numbers, website links, social handles, zip codes, and street/city references.
    """
    if not isinstance(text, str):
        return ""

    # First, strip leading names up to street keywords or contact patterns
    # Matches letters and spaces at the start, followed by street keywords
    text = re.sub(
        r'^[a-z\s]+?\b(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|way|court|ct|loop|plaza|pkwy|parkway)\b',
        '',
        text,
        flags=re.IGNORECASE
    )
    
    # Matches letters and spaces at the start, followed by contact details (phone, link, email, etc.)
    text = re.sub(
        r'^[a-z\s]+?\b(?:\d{10,12}|linkedincom\S*|githubcom\S*|twittercom\S*|resumesampleexamplecom|gmailcom|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Common email regex
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    # Phone numbers (various formats, including spaces and dashes)
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
    
    # Links/URLs
    url_pattern = r'\b(?:https?://)?(?:www\.)?(?:linkedin\.com/in/|github\.com/|twitter\.com/|resumesampleexample\.com)[a-zA-Z0-9._%+-/]*\b'
    # Shortened variants in OCR text
    ocr_links = r'\b(?:linkedincomin|githubcom|twittercom|resumesampleexamplecom)\S*\b'
    
    # Address and zip codes
    zip_pattern = r'\b\d{5}(?:-\d{4})?\b'
    address_keywords = r'\b[a-zA-Z0-9\s,.-]{1,60}\b(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|way|court|ct|loop|plaza|pkwy|parkway)\b'
    city_state_pattern = r'\b[A-Z][a-zA-Z\s,.-]{1,20}\s+[A-Z]{2}\b' # e.g. "San Francisco CA" or "Tracy CA"
    
    # Compile all patterns
    text = re.sub(email_pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(phone_pattern, '', text)
    text = re.sub(url_pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(ocr_links, '', text, flags=re.IGNORECASE)
    text = re.sub(address_keywords, '', text, flags=re.IGNORECASE)
    text = re.sub(city_state_pattern, '', text)
    text = re.sub(zip_pattern, '', text)
    
    # OCR-style noise patterns: sequences of numbers with long runs of spaces or tabs
    # (often found in the phone number/contact block of the raw dataset)
    text = re.sub(r'\b\d{5}\s+\d{3}\s+\d{4,7}\b', '', text)
    text = re.sub(r'\b\d{5,10}\b', '', text)
    
    # Let's clean up text that starts with common header words (e.g. "jessica claire montgomery street san francisco ca")
    # If the text starts with a personal name or typical placeholder header, let's remove common leak prefixes
    header_keywords = [
        r'^name\s*:\s*[a-zA-Z\s]+',
        r'^phone\s*:\s*[\d\s+-]+',
        r'^email\s*:\s*\S+',
        r'^address\s*:\s*[a-zA-Z0-9\s,.-]+',
        r'^contact\s*:\s*[a-zA-Z0-9\s,.-]+'
    ]
    for hk in header_keywords:
        text = re.sub(hk, '', text, flags=re.IGNORECASE)
        
    return clean_whitespace(text)

def clean_ocr_text(text: str) -> str:
    """
    Cleans OCR-style concatenated text, digit runs, and run-together words.
    """
    if not isinstance(text, str):
        return ""
    
    # Strip contact boilerplate first
    text = strip_contact_boilerplate(text)
    
    # Replace weird formatting, non-ascii characters (or normalize them)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Remove weird bullet points and isolated characters (like single characters or digits floating around)
    # But preserve meaningful numbers and letters.
    text = re.sub(r'\b[a-zA-Z]\b', '', text) # remove single letters (like 'a' or 'i' if floating, but regex matches isolated single letters)
    # Actually, keep "a" and "I" if they are part of English, but regex-based cleaning of OCR artifacts:
    # Remove characters like |, •, *, _ etc.
    text = re.sub(r'[\x00-\x1F\x7F-\x9F\|•\*_#]', ' ', text)
    
    return clean_whitespace(text)

def deduplicate_resumes(df: pd.DataFrame, text_col: str = 'Text', threshold: float = 0.95) -> pd.DataFrame:
    """
    De-duplicates near-identical resumes using TF-IDF and Cosine Similarity.
    Keeps the first occurrence of each unique resume.
    """
    if df.empty:
        return df
    
    # Fill missing values
    texts = df[text_col].fillna("").tolist()
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Compute Cosine Similarity
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Find duplicates
    n = len(df)
    to_keep = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not to_keep[i]:
            continue
        # Find all other resumes with similarity above threshold
        similar_indices = np.where(sim_matrix[i] > threshold)[0]
        for idx in similar_indices:
            if idx > i:
                to_keep[idx] = False
                
    return df[to_keep].reset_index(drop=True)
