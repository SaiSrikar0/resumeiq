import re
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# PART 1: Text Cleaning (formerly cleaner.py)
# =====================================================================

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
    text = re.sub(r'\b[a-zA-Z]\b', '', text) # remove single letters
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


# =====================================================================
# PART 2: Entity Extraction (formerly extractor.py)
# =====================================================================

# Master list of skills mapped to standardized representations
SKILL_PATTERNS = {
    # Programming Languages
    "Python": [r"\bpython\b"],
    "Java": [r"\bjava\b(?!script)"],
    "JavaScript": [r"\bjavascript\b", r"\bjs\b"],
    "TypeScript": [r"\btypescript\b", r"\bts\b"],
    "C++": [r"\bc\+\+\b", r"\bcpp\b"],
    "C#": [r"\bc#\b", r"\bcsharp\b"],
    "C": [r"\b(?<![pP]&)(?<![pP]\s&)(?<![pP]\s&\s)[cC]\s+programming\b", r"\blanguage\s+[cC]\b", r"\b[cC]\s+language\b", r"\b[cC]\s+coding\b"],
    "Go": [r"\bgo\b(?!lang)", r"\bgolang\b"],
    "Ruby": [r"\bruby\b"],
    "PHP": [r"\bphp\b"],
    "Rust": [r"\brust\b"],
    "Swift": [r"\bswift\b"],
    "Kotlin": [r"\bkotlin\b"],
    "Objective-C": [r"\bobjective-c\b", r"\bobjc\b"],
    "SQL": [r"\bsql\b"],
    "Scala": [r"\bscala\b"],
    "R": [r"\br\b(?![#\+])"],
    "HTML": [r"\bhtml5?\b"],
    "CSS": [r"\bcss3?\b"],
    "Shell/Bash": [r"\bbash\b", r"\bshell\s+script(?:ing)?\b"],
    "PowerShell": [r"\bpowershell\b"],
    
    # Core Fundamentals
    "OOP": [r"\boop\b", r"\boops\b", r"\bobject[- ]oriented\b", r"\bobject[- ]oriented\s+programming\b"],
    "Data Structures": [r"\bdata\s+structures?\b", r"\bdsa\b", r"\balgorithms?\b"],
    "REST APIs": [r"\brest\s*apis?\b", r"\brestful\b", r"\brest\b(?!\s+area)"],
    "Web Applications": [r"\bweb\s+apps?\b", r"\bweb\s+applications?\b", r"\bweb\s+development\b", r"\bweb\s+dev\b", r"\bfull-?stack\b"],
    "Database": [r"\bdatabase(?:s|\s+management|\s+systems?)?\b", r"\bdbms\b"],

    # Frontend
    "React": [r"\breact\b", r"\breact\.js\b", r"\breactjs\b"],
    "Angular": [r"\bangular\b", r"\bangularjs\b"],
    "Vue": [r"\bvue\b", r"\bvuejs\b"],
    "Next.js": [r"\bnext\.js\b", r"\bnextjs\b"],
    "Redux": [r"\bredux\b"],
    "Tailwind": [r"\btailwind\b", r"\btailwindcss\b"],
    "Sass": [r"\bsass\b", r"\bscss\b"],
    "jQuery": [r"\bjquery\b"],
    "Bootstrap": [r"\bbootstrap\b"],
    "Streamlit": [r"\bstreamlit\b"],
    
    # Backend
    "Node.js": [r"\bnode\.js\b", r"\bnodejs\b", r"\bnode\b"],
    "Express": [r"\bexpress\.js\b", r"\bexpressjs\b", r"\bexpress\b(?!\s+delivery)"],
    "Django": [r"\bdjango\b"],
    "Flask": [r"\bflask\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Spring Boot": [r"\bspring\s+boot\b", r"\bspring\b"],
    "ASP.NET": [r"\basp\.net\b", r"\bdotnet\b", r"\b\.net\b"],
    "Laravel": [r"\blaravel\b"],
    "Rails": [r"\brails\b", r"\bruby\s+on\s+rails\b"],
    
    # Databases / Big Data
    "MySQL": [r"\bmysql\b"],
    "PostgreSQL": [r"\bpostgresql\b", r"\bpostgres\b", r"\bsupabase\b"],
    "Oracle": [r"\boracle\b"],
    "SQL Server": [r"\bsql\s*server\b", r"\bmssql\b"],
    "SQLite": [r"\bsqlite\b"],
    "MongoDB": [r"\bmongodb\b", r"\bmongo\b"],
    "Redis": [r"\bredis\b"],
    "Cassandra": [r"\bcassandra\b"],
    "DynamoDB": [r"\bdynamodb\b"],
    "Elasticsearch": [r"\belasticsearch\b", r"\belastic\b"],
    "Firebase": [r"\bfirebase\b"],
    "Spark": [r"\bspark\b", r"\bapache\s+spark\b"],
    "Hadoop": [r"\bhadoop\b"],
    "ETL": [r"\betl\b"],
    "Informatica": [r"\binformatica\b"],
    "Talend": [r"\btalend\b"],
    "SSIS": [r"\bssis\b"],
    "Data Warehouse": [r"\bdata\s+warehous(?:e|ing)\b", r"\bdwh\b"],
    
    # Cloud / DevOps
    "AWS": [r"\baws\b", r"\bamazon\s+web\s+services\b"],
    "Azure": [r"\bazure\b"],
    "GCP": [r"\bgcp\b", r"\bgoogle\s+cloud\b"],
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "Ansible": [r"\bansible\b"],
    "Terraform": [r"\bterraform\b"],
    "Jenkins": [r"\bjenkins\b"],
    "Git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"],
    "CI/CD": [r"\bci\s*/\s*cd\b", r"\bci-cd\b", r"\bcontinuous\s+integration\b"],
    "Linux": [r"\blinux\b", r"\bunix\b", r"\bubuntu\b", r"\bcentos\b", r"\bredhat\b"],
    "Nginx": [r"\bnginx\b"],
    "Apache": [r"\bapache\b"],
    "SRE": [r"\bsre\b", r"\bsite\s+reliability\s+engineering\b"],
    "Prometheus": [r"\bprometheus\b"],
    "Grafana": [r"\bgrafana\b"],
    
    # AI / Data Science
    "Generative AI": [r"\bgen-?ai\b", r"\bgenerative\s+ai\b", r"\bllms?\b", r"\bollama\b", r"\bgroq\b", r"\bdiffusion\s+models?\b"],
    "Machine Learning": [r"\bmachine\s+learning\b", r"\bml\b"],
    "Deep Learning": [r"\bdeep\s+learning\b", r"\bdl\b"],
    "Artificial Intelligence": [r"\bartificial\s+intelligence\b", r"\bai\b"],
    "NLP": [r"\bnlp\b", r"\bnatural\s+language\s+processing\b"],
    "Computer Vision": [r"\bcomputer\s+vision\b", r"\bcv\b"],
    "PyTorch": [r"\bpytorch\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "Keras": [r"\bkeras\b"],
    "Scikit-learn": [r"\bscikit-learn\b", r"\bsklearn\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Tableau": [r"\btableau\b"],
    "PowerBI": [r"\bpowerbi\b", r"\bpower\s+bi\b"],
    "Data Analytics": [r"\beda\b", r"\bdata\s+analytics?\b", r"\bdata\s+analysis\b", r"\bexploratory\s+data\s+analysis\b", r"\bmatplotlib\b", r"\bseaborn\b"],
    "Anomaly Detection": [r"\banomaly\s+detection\b", r"\bisolation\s+forest\b"],
    
    # Security
    "Network Security": [r"\bnetwork\s+security\b"],
    "Cybersecurity": [r"\bcybersecurity\b", r"\bcyber\s+security\b"],
    "Penetration Testing": [r"\bpen(?:etration)?\s+test(?:ing)?\b"],
    "Cryptography": [r"\bcryptography\b"],
    "Firewall": [r"\bfirewall\b"],
    
    # Management / Methodology
    "Business Analysis": [r"\bbusiness\s+analysis\b", r"\bba\b(?!script)"],
    "Agile": [r"\bagile\b"],
    "Scrum": [r"\bscrum\b"],
    "Kanban": [r"\bkanban\b"],
    "JIRA": [r"\bjira\b"],
    "Product Management": [r"\bproduct\s+management\b"],
    "Project Management": [r"\bproject\s+management\b", r"\bpmp\b"],
    "SDLC": [r"\bsdlc\b"],
    "Business Intelligence": [r"\bbusiness\s+intelligence\b", r"\bbi\b"],
    
    # UI/UX & Design
    "UI/UX": [r"\bui\s*/\s*ux\b", r"\bui-ux\b", r"\buser\s+interface\b"],
    "Figma": [r"\bfigma\b"],
    "Adobe XD": [r"\badobe\s+xd\b"],
    "Photoshop": [r"\bphotoshop\b"],
    
    # SAP
    "SAP": [r"\bsap\b"],
    "ABAP": [r"\babap\b"],
    
    # Blockchain
    "Blockchain": [r"\bblockchain\b"],
    "Solidity": [r"\bsolidity\b"],
    "Smart Contracts": [r"\bsmart\s+contracts?\b"],
    
    # Mobile
    "Android": [r"\bandroid\b"],
    "iOS": [r"\bios\b"],
    "React Native": [r"\breact\s+native\b"],
    "Flutter": [r"\bflutter\b"],
    
    # QA / Testing
    "QA": [r"\bqa\b", r"\bquality\s+assurance\b"],
    "Testing": [r"\btesting\b", r"\btest\s+cases?\b", r"\btesting\s+lifecycle\b"],
    "Selenium": [r"\bselenium\b"],
    
    # Technical Writing
    "Technical Writing": [r"\btechnical\s+writing\b", r"\bdocumentation\b"]
}

def extract_skills(text: str) -> List[str]:
    """Extract skills from text based on regex mapping."""
    if not isinstance(text, str):
        return []
    
    found_skills = []
    text_lower = text.lower()
    
    for skill, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            # Check for pattern match in lowercased text
            if re.search(pattern, text_lower):
                found_skills.append(skill)
                break # Only need one pattern to match this skill
                
    return found_skills

def parse_date_ranges(text: str) -> float:
    """
    Parses date intervals from OCR-style concatenated text and calculates total years.
    """
    if not isinstance(text, str):
        return 0.0
    
    total_years = 0.0
    current_year = 2026 # Normalized current year
    
    # 1. OCR pattern for MMYYYY MMYYYY (e.g. 102011 112013)
    ocr_mmyyyy_mmyyyy = re.findall(r'\b(\d{2})(\d{4})\s+(\d{2})(\d{4})\b', text)
    for m1, y1, m2, y2 in ocr_mmyyyy_mmyyyy:
        y1, y2 = int(y1), int(y2)
        m1, m2 = int(m1), int(m2)
        if 1970 <= y1 <= current_year and 1970 <= y2 <= current_year and 1 <= m1 <= 12 and 1 <= m2 <= 12:
            duration = (y2 - y1) + (m2 - m1) / 12.0
            if 0 < duration < 15:
                total_years += duration
                
    # 2. OCR pattern for MMYYYY YYYY (e.g. 052005 2007)
    ocr_mmyyyy_yyyy = re.findall(r'\b(\d{2})(\d{4})\s+(\d{4})\b', text)
    for m1, y1, y2 in ocr_mmyyyy_yyyy:
        y1, y2 = int(y1), int(y2)
        m1 = int(m1)
        if 1970 <= y1 <= current_year and 1970 <= y2 <= current_year and 1 <= m1 <= 12:
            duration = (y2 - y1) + (6 - m1) / 12.0 # assume June if second month missing
            if 0 < duration < 15:
                total_years += duration
                
    # 3. Standard YYYY - YYYY or YYYY to YYYY or YYYY - current
    standard_yyyy_yyyy = re.findall(r'\b(\d{4})\s*(?:-|to|till)\s*(\d{4}|current|present)\b', text, flags=re.IGNORECASE)
    for y1, y2 in standard_yyyy_yyyy:
        y1 = int(y1)
        if y2.lower() in ['current', 'present']:
            y2 = current_year
        else:
            y2 = int(y2)
        if 1970 <= y1 <= current_year and 1970 <= y2 <= current_year:
            duration = float(y2 - y1)
            if 0 < duration < 15:
                total_years += duration
                
    # 3.5. OCR YYYY current/present (space-separated)
    ocr_yyyy_current = re.findall(r'\b(\d{4})\s+(current|present)\b', text, flags=re.IGNORECASE)
    for y1, y2 in ocr_yyyy_current:
        y1 = int(y1)
        y2 = current_year
        if 1970 <= y1 <= current_year:
            duration = float(y2 - y1)
            if 0 < duration < 15:
                total_years += duration

    # 4. Standard YYYY YYYY (OCR-style with spaces)
    ocr_yyyy_yyyy = re.findall(r'\b(\d{4})\s+(\d{4})\b', text)
    for y1, y2 in ocr_yyyy_yyyy:
        y1, y2 = int(y1), int(y2)
        if 1970 <= y1 <= current_year and 1970 <= y2 <= current_year and y2 >= y1:
            duration = float(y2 - y1)
            if 0 < duration < 10: # Cap per interval to prevent overlaps
                total_years += duration
                
    return min(total_years, 40.0) # Cap total years at 40 to avoid OCR anomalies

def extract_years_experience(text: str, experience_text: str = None) -> float:
    """
    Extracts years of experience using:
    1. Direct phrases: e.g. "bringing 10 years software", "3 years experience", "10+ years".
    2. Sum of parsed role date ranges in the experience section.
    Returns the maximum value between the two heuristics.
    """
    if not isinstance(text, str):
        return 0.0
    
    direct_mentions = []
    patterns = [
        r'\b(\d{1,2})\+?\s*(?:years?|yrs?)\s+(?:[a-zA-Z ]{1,30}\s+)?(?:of\s+)?experience\b',
        r'\bexperience\s*:\s*(\d{1,2})\+?\s*(?:years?|yrs?)\b',
        r'\bbringing\s+(\d{1,2})\s+(?:of\s+)?years?\b',
        r'\b(\d{1,2})\+?\s*(?:years?|yrs?)\s+in\s+[a-zA-Z\s]{1,15}\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for val in matches:
            direct_mentions.append(float(val))
            
    direct_val = max(direct_mentions) if direct_mentions else 0.0
    
    eval_text = experience_text if experience_text else text
    parsed_val = parse_date_ranges(eval_text)
    
    res = max(direct_val, parsed_val)
    return round(res, 1)

def extract_degree(text: str, is_jd: bool = False) -> str:
    """
    Extracts degree level mentioned in text.
    For resumes: PhD > Master's > Bachelor's > None
    For JDs (is_jd=True): returns minimum acceptable qualification (Bachelor's > Master's > PhD > None)
    """
    if not isinstance(text, str):
        return "None"
    
    text_lower = text.lower()
    phd_pattern = r'\b(?:phd|ph\.d|doctorate|doctor\s+of\s+philosophy)\b'
    masters_pattern = r'\b(?:m\.?\s*tech|m\.?\s*e|m\.?\s*s|m\.?\s*sc|m\.?\s*a|m\.?\s*ca|m\.?\s*ba|master\'?s?|postgraduate)\b'
    bachelors_pattern = r'\b(?:b\.?\s*tech|b\.?\s*e|b\.?\s*s|b\.?\s*sc|b\.?\s*a|b\.?\s*ca|bachelor\'?s?|undergraduate)\b'
    
    if is_jd:
        if re.search(bachelors_pattern, text_lower):
            return "Bachelor's"
        elif re.search(masters_pattern, text_lower):
            return "Master's"
        elif re.search(phd_pattern, text_lower):
            return "PhD"
        return "None"
    else:
        if re.search(phd_pattern, text_lower):
            return "PhD"
        elif re.search(masters_pattern, text_lower) and not re.search(bachelors_pattern, text_lower):
            return "Master's"
        elif re.search(bachelors_pattern, text_lower):
            return "Bachelor's"
        elif re.search(masters_pattern, text_lower):
            return "Master's"
        else:
            return "None"

def extract_structured_fields(row: Dict) -> Dict:
    """
    Processes a raw resume row and extracts structured skills, years of experience, and degree level.
    """
    text = row.get("Text", "")
    exp_text = row.get("Experience", "")
    edu_text = row.get("Education", "")
    
    skills = extract_skills(text)
    years_exp = extract_years_experience(text, experience_text=exp_text)
    degree = extract_degree(edu_text) if edu_text else "None"
    if degree == "None":
        degree = extract_degree(text)
        
    return {
        "ExtractedSkills": skills,
        "YearsOfExperience": years_exp,
        "DegreeLevel": degree
    }
