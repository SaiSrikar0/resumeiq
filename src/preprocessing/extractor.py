import re
from typing import List, Dict, Set, Tuple

# Master list of skills mapped to standardized representations
SKILL_PATTERNS = {
    # Programming Languages
    "Python": [r"\bpython\b"],
    "Java": [r"\bjava\b(?!script)"],
    "JavaScript": [r"\bjavascript\b", r"\bjs\b"],
    "TypeScript": [r"\btypescript\b", r"\bts\b"],
    "C++": [r"\bc\+\+\b", r"\bcpp\b"],
    "C#": [r"\bc#\b", r"\bcsharp\b"],
    "C": [r"\bc\b(?![#\+])"],
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
    "PostgreSQL": [r"\bpostgresql\b", r"\bpostgres\b"],
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
    "Testing": [r"\btesting\b", r"\btest\s+cases?\b"],
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
    Examples of OCR run-together format:
      "102011 112013" -> Oct 2011 to Nov 2013 (approx 2 years)
      "092008 102011" -> Sept 2008 to Oct 2011 (approx 3 years)
      "052005 2007" -> May 2005 to 2007 (approx 2 years)
      "2014 current" -> 2014 to present (approx 12 years if 2026)
      "2014 2018" -> 2014 to 2018 (4 years)
    """
    if not isinstance(text, str):
        return 0.0
    
    total_years = 0.0
    
    # Normalize current year
    current_year = 2026 # Context says local time is 2026-07-19
    
    # 1. OCR pattern for MMYYYY MMYYYY (e.g. 102011 112013)
    # Match sequences of two 6-digit numbers representing MMYYYY
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
                
    # 3. Standard YYYY - YYYY or YYYY to YYYY or YYYY - current (with separator)
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
                
    # 3.5. OCR YYYY current/present (space-separated, no standard separator)
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
    # filter out already parsed ones by ensuring we don't count duplicate spans if they overlap
    # but for simple heuristic:
    for y1, y2 in ocr_yyyy_yyyy:
        y1, y2 = int(y1), int(y2)
        if 1970 <= y1 <= current_year and 1970 <= y2 <= current_year and y2 >= y1:
            duration = float(y2 - y1)
            if 0 < duration < 10: # Cap at 10 years per interval to prevent overlaps and weird spans
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
    
    # Heuristic 1: Regex on direct mentions in Text
    direct_mentions = []
    # Match phrases like "10 years experience", "3+ years of experience", "bringing 10 years"
    patterns = [
        r'\b(\d{1,2})\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience\b',
        r'\bexperience\s*:\s*(\d{1,2})\+?\s*(?:years?|yrs?)\b',
        r'\bbringing\s+(\d{1,2})\s+(?:of\s+)?years?\b',
        r'\b(\d{1,2})\+?\s*(?:years?|yrs?)\s+in\s+[a-zA-Z\s]{1,15}\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for val in matches:
            direct_mentions.append(float(val))
            
    direct_val = max(direct_mentions) if direct_mentions else 0.0
    
    # Heuristic 2: Sum date ranges
    # Use the experience text if provided, otherwise fallback to entire text
    eval_text = experience_text if experience_text else text
    parsed_val = parse_date_ranges(eval_text)
    
    # Return max, rounded to 1 decimal place
    res = max(direct_val, parsed_val)
    return round(res, 1)

def extract_degree(text: str) -> str:
    """
    Extracts the highest degree level mentioned in text.
    Mapping priority: PhD > Master's > Bachelor's > Associate/Other > None
    """
    if not isinstance(text, str):
        return "None"
    
    text_lower = text.lower()
    
    # PhD pattern
    phd_pattern = r'\b(?:phd\b|ph\.d\.(?!\w)|doctorate\b|doctor\s+of\s+philosophy\b)'
    # Master's pattern
    masters_pattern = r'\b(?:ms\b|m\.s\.(?!\w)|msc\b|m\.sc\.(?!\w)|mtech\b|m\.tech\.(?!\w)|mba\b|m\.b\.a\.(?!\w)|master\b|masters\b|postgraduate\b)'
    # Bachelor's pattern
    bachelors_pattern = r'\b(?:bs\b|b\.s\.(?!\w)|bsc\b|b\.sc\.(?!\w)|btech\b|b\.tech\.(?!\w)|ba\b|b\.a\.(?!\w)|bachelor\b|bachelors\b|undergraduate\b|graduate\b)'
    
    if re.search(phd_pattern, text_lower):
        return "PhD"
    elif re.search(masters_pattern, text_lower):
        return "Master's"
    elif re.search(bachelors_pattern, text_lower):
        return "Bachelor's"
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
    
    # If the raw "Skills" field in the dataset is actually meaningful (e.g. not a placeholder "Python, SQL, Git, Linux"), 
    # we can combine it, but from what we saw it contains placeholders. We will trust our extracted skills.
    
    return {
        "ExtractedSkills": skills,
        "YearsOfExperience": years_exp,
        "DegreeLevel": degree
    }
