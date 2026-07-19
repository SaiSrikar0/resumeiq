import json
import os

SEED_JOB_DESCRIPTIONS = [
    {
        "JobDescriptionID": "JD_0001",
        "Category": "Java Developer",
        "Title": "Senior Java Developer",
        "RequiredSkills": ["Java", "Spring Boot", "SQL", "Git", "Docker", "CI/CD"],
        "RequiredExperience": 5.0,
        "RequiredEducation": "Bachelor's",
        "Description": "We are seeking a Senior Java Developer to design, develop, and maintain high-performance enterprise applications. You will work with Spring Boot microservices, SQL databases, and deploy systems in containerized environments using Docker and CI/CD pipelines."
    },
    {
        "JobDescriptionID": "JD_0002",
        "Category": "Java Developer",
        "Title": "Junior Java Developer",
        "RequiredSkills": ["Java", "SQL", "HTML", "CSS", "Git"],
        "RequiredExperience": 1.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Looking for a Junior Java Developer to assist in developing web applications and database integrations. Requires familiarity with Core Java, SQL queries, and basic frontend technologies like HTML/CSS."
    },
    {
        "JobDescriptionID": "JD_0003",
        "Category": "Python Developer",
        "Title": "Python Backend Engineer",
        "RequiredSkills": ["Python", "Django", "Flask", "FastAPI", "PostgreSQL", "Git", "Docker"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Join our backend team building REST APIs and microservices. Strong Python skills required along with Django/Flask/FastAPI, database query optimization (PostgreSQL), and containerization."
    },
    {
        "JobDescriptionID": "JD_0004",
        "Category": "Python Developer",
        "Title": "Lead Python Developer",
        "RequiredSkills": ["Python", "Django", "AWS", "Docker", "Kubernetes", "CI/CD", "PostgreSQL"],
        "RequiredExperience": 7.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Lead a team of Python engineers building scalable backend services on AWS. Responsibilities include system architecture, mentoring junior engineers, and managing deployments with Docker and Kubernetes."
    },
    {
        "JobDescriptionID": "JD_0005",
        "Category": "Data Science",
        "Title": "Data Scientist",
        "RequiredSkills": ["Python", "SQL", "Machine Learning", "Scikit-learn", "Pandas", "NumPy", "Tableau"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Master's",
        "Description": "Seeking a Data Scientist to analyze complex datasets, build predictive models, and build dashboards in Tableau. Strong scripting in Python, data cleaning with Pandas/NumPy, and model training with Scikit-learn required."
    },
    {
        "JobDescriptionID": "JD_0006",
        "Category": "Data Science",
        "Title": "Senior Data Scientist",
        "RequiredSkills": ["Python", "SQL", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "Spark"],
        "RequiredExperience": 6.0,
        "RequiredEducation": "PhD",
        "Description": "Design and build production-grade machine learning and deep learning pipelines. Work with PyTorch/TensorFlow, SQL databases, and big data tools like Apache Spark to deliver business-critical intelligence."
    },
    {
        "JobDescriptionID": "JD_0007",
        "Category": "DevOps",
        "Title": "DevOps Engineer",
        "RequiredSkills": ["Docker", "Kubernetes", "AWS", "Terraform", "Jenkins", "Linux", "Shell/Bash", "CI/CD"],
        "RequiredExperience": 4.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Manage our cloud infrastructure and deployment pipelines. Implement Infrastructure as Code (IaC) with Terraform, manage Kubernetes clusters on AWS, and automate CI/CD processes."
    },
    {
        "JobDescriptionID": "JD_0008",
        "Category": "SQL Developer",
        "Title": "Database/SQL Developer",
        "RequiredSkills": ["SQL", "SQL Server", "MySQL", "PostgreSQL", "ETL", "SSIS"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Design and optimize databases, write stored procedures, and develop ETL pipelines using SSIS. Strong SQL querying and schema design skills are essential."
    },
    {
        "JobDescriptionID": "JD_0009",
        "Category": "Database",
        "Title": "Database Engineer",
        "RequiredSkills": ["SQL", "Oracle", "SQL Server", "MySQL", "PostgreSQL", "Database Administrator"],
        "RequiredExperience": 4.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Maintain database integrity, design relational models, and optimize performance across multiple database instances including Oracle and MySQL."
    },
    {
        "JobDescriptionID": "JD_0010",
        "Category": "Testing",
        "Title": "QA Automation Engineer",
        "RequiredSkills": ["QA", "Testing", "Selenium", "Java", "Python", "Git"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Develop automated testing scripts for web applications using Selenium. Write test plans, execute regression runs, and track defects in collaboration with developers."
    },
    {
        "JobDescriptionID": "JD_0011",
        "Category": "Web Designing",
        "Title": "Web Designer",
        "RequiredSkills": ["HTML", "CSS", "UI/UX", "Figma", "Photoshop", "Bootstrap"],
        "RequiredExperience": 2.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Design and code web layouts and templates. Work with UI/UX designs in Figma and translate them into responsive HTML/CSS using Bootstrap."
    },
    {
        "JobDescriptionID": "JD_0012",
        "Category": "React Developer",
        "Title": "React Frontend Engineer",
        "RequiredSkills": ["React", "JavaScript", "TypeScript", "Redux", "HTML", "CSS", "Tailwind"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Build modern web applications using React. Expert level knowledge of JavaScript/TypeScript, state management with Redux, and layouts with Tailwind CSS."
    },
    {
        "JobDescriptionID": "JD_0013",
        "Category": "Business Analyst",
        "Title": "Business Analyst",
        "RequiredSkills": ["Business Analysis", "Requirements Gathering", "Agile", "Scrum", "JIRA", "SQL"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Act as a bridge between business stakeholders and tech teams. Gather and document requirements, manage JIRA boards in an Agile/Scrum environment, and write basic SQL queries to retrieve data."
    },
    {
        "JobDescriptionID": "JD_0014",
        "Category": "DotNet Developer",
        "Title": "C#/.NET Developer",
        "RequiredSkills": ["C#", "ASP.NET", "SQL Server", "Git", "JavaScript"],
        "RequiredExperience": 4.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Develop backend services and web applications using C# and ASP.NET Core. Integrate SQL Server database layers and collaborate with frontend developers."
    },
    {
        "JobDescriptionID": "JD_0015",
        "Category": "Software Developer",
        "Title": "Software Developer",
        "RequiredSkills": ["Java", "Python", "C++", "SQL", "Git", "SDLC"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "General Software Developer to work on various applications and scripts. Requires experience with Java or Python, git version control, and understanding of the SDLC."
    },
    {
        "JobDescriptionID": "JD_0016",
        "Category": "ETL Developer",
        "Title": "ETL Data Integration Engineer",
        "RequiredSkills": ["ETL", "SQL", "Informatica", "Talend", "Data Warehouse", "SSIS"],
        "RequiredExperience": 4.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Build data pipelines and warehousing systems. Strong ETL skills with Informatica or Talend and advanced database programming (SQL)."
    },
    {
        "JobDescriptionID": "JD_0017",
        "Category": "Network Security Engineer",
        "Title": "Network Security Engineer",
        "RequiredSkills": ["Network Security", "Cybersecurity", "Firewall", "Linux", "AWS"],
        "RequiredExperience": 5.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Secure our corporate network and cloud assets. Configure firewalls, audit network security parameters, and analyze threat landscapes in Linux/AWS environments."
    },
    {
        "JobDescriptionID": "JD_0018",
        "Category": "Full Stack Developer",
        "Title": "Full Stack Engineer (Node + React)",
        "RequiredSkills": ["React", "Node.js", "Express", "JavaScript", "TypeScript", "MongoDB", "CSS", "Git"],
        "RequiredExperience": 4.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Build and scale full-stack web applications. Strong React skills for frontend and Node/Express backend, working with MongoDB database."
    },
    {
        "JobDescriptionID": "JD_0019",
        "Category": "SAP Developer",
        "Title": "SAP ABAP Consultant",
        "RequiredSkills": ["SAP", "ABAP", "SQL"],
        "RequiredExperience": 4.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Configure SAP installations and write custom ABAP code to meet specific business workflow requirements."
    },
    {
        "JobDescriptionID": "JD_0020",
        "Category": "Digital Media",
        "Title": "Digital Media Specialist",
        "RequiredSkills": ["Adobe XD", "Photoshop", "UI/UX", "Figma", "Technical Writing"],
        "RequiredExperience": 2.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Create digital content, mockups, and assets. Graphic design skills using Photoshop and Figma combined with web aesthetics layout knowledge."
    },
    {
        "JobDescriptionID": "JD_0021",
        "Category": "Cloud Engineer",
        "Title": "Cloud Infrastructure Engineer",
        "RequiredSkills": ["AWS", "Azure", "Terraform", "Docker", "Kubernetes", "Linux"],
        "RequiredExperience": 4.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Build and scale cloud environments across AWS and Azure. Automate deployments using Terraform and manage container runtime environments."
    },
    {
        "JobDescriptionID": "JD_0022",
        "Category": "Machine Learning Engineer",
        "Title": "Machine Learning Engineer",
        "RequiredSkills": ["Python", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "Scikit-learn", "Docker"],
        "RequiredExperience": 4.0,
        "RequiredEducation": "Master's",
        "Description": "Develop, train, and deploy machine learning models. Expertise in PyTorch, TensorFlow, Scikit-learn, and containerizing ML microservices using Docker is required."
    },
    {
        "JobDescriptionID": "JD_0023",
        "Category": "Frontend Developer",
        "Title": "Frontend UI Developer",
        "RequiredSkills": ["HTML", "CSS", "JavaScript", "React", "Tailwind", "Bootstrap"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Design and build beautiful user interfaces. Requires proficiency in JavaScript, React, CSS grids, and Tailwind CSS."
    },
    {
        "JobDescriptionID": "JD_0024",
        "Category": "Backend Developer",
        "Title": "Backend API Developer",
        "RequiredSkills": ["Python", "FastAPI", "SQL", "PostgreSQL", "Redis", "Docker"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Develop backend REST APIs using Python and FastAPI. Optimize SQL databases (PostgreSQL) and implement caching mechanisms using Redis."
    },
    {
        "JobDescriptionID": "JD_0025",
        "Category": "AI Engineer",
        "Title": "Generative AI Engineer",
        "RequiredSkills": ["Python", "Artificial Intelligence", "NLP", "Deep Learning", "PyTorch", "transformers"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Master's",
        "Description": "Implement conversational AI models and RAG pipelines. Strong knowledge of HuggingFace transformers, PyTorch, and large language model tuning."
    },
    {
        "JobDescriptionID": "JD_0026",
        "Category": "Cybersecurity Analyst",
        "Title": "Cybersecurity Analyst",
        "RequiredSkills": ["Cybersecurity", "Network Security", "Linux", "Firewall", "Penetration Testing"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Monitor systems for security anomalies, audit cloud configurations, conduct penetration testing, and recommend fixes."
    },
    {
        "JobDescriptionID": "JD_0027",
        "Category": "QA Engineer",
        "Title": "QA Engineer (Manual & Automated)",
        "RequiredSkills": ["QA", "Testing", "Selenium", "Git"],
        "RequiredExperience": 2.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Write and execute test cases, identify bugs, and build test automation workflows using Selenium."
    },
    {
        "JobDescriptionID": "JD_0028",
        "Category": "Database Administrator",
        "Title": "Database Administrator (DBA)",
        "RequiredSkills": ["Database Administrator", "SQL", "Oracle", "SQL Server", "Linux"],
        "RequiredExperience": 5.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Ensure database availability, back up systems, perform upgrades, and tune queries for PostgreSQL, Oracle, and SQL Server."
    },
    {
        "JobDescriptionID": "JD_0029",
        "Category": "UI/UX Designer",
        "Title": "Product UI/UX Designer",
        "RequiredSkills": ["UI/UX", "Figma", "Adobe XD", "Wireframing"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Design user flows, mockups, and interactive prototypes. Expert skills in Figma and visual design patterns are required."
    },
    {
        "JobDescriptionID": "JD_0030",
        "Category": "Blockchain",
        "Title": "Blockchain Engineer",
        "RequiredSkills": ["Blockchain", "Solidity", "Smart Contracts", "Cryptography", "Git"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Develop smart contracts and distributed applications. Requires Solidity development skills and understanding of decentralized protocols."
    },
    {
        "JobDescriptionID": "JD_0031",
        "Category": "Site Reliability Engineer",
        "Title": "Site Reliability Engineer (SRE)",
        "RequiredSkills": ["SRE", "Linux", "Docker", "Kubernetes", "Prometheus", "Grafana", "Python", "CI/CD"],
        "RequiredExperience": 4.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Focus on reliability, scalability, and performance of services. Automate operations using Python/Bash, monitor with Prometheus/Grafana, and debug Linux systems."
    },
    {
        "JobDescriptionID": "JD_0032",
        "Category": "Mobile Developer",
        "Title": "Mobile App Developer (Flutter/React Native)",
        "RequiredSkills": ["Android", "iOS", "Flutter", "React Native", "Git"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Develop cross-platform mobile apps for Android and iOS using Flutter or React Native."
    },
    {
        "JobDescriptionID": "JD_0033",
        "Category": "System Administrator",
        "Title": "Linux/Windows System Administrator",
        "RequiredSkills": ["Linux", "Windows Server", "Active Directory", "Shell/Bash", "PowerShell", "Network Security"],
        "RequiredExperience": 4.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Administer directory services, manage server deployments, deploy security patches, and automate tasks via PowerShell/Bash."
    },
    {
        "JobDescriptionID": "JD_0034",
        "Category": "Technical Lead",
        "Title": "Technical Lead",
        "RequiredSkills": ["Java", "Python", "Git", "Agile", "Scrum", "SDLC", "System Design"],
        "RequiredExperience": 8.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Provide technical leadership, design architecture, mentor developers, and lead project delivery in an Agile environment."
    },
    {
        "JobDescriptionID": "JD_0035",
        "Category": "Blockchain Developer",
        "Title": "Smart Contract Developer",
        "RequiredSkills": ["Blockchain", "Solidity", "Smart Contracts", "Git"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Write and audit Solidity code for ERC20 and ERC721 smart contracts. Deploy contracts on Ethereum or layer-2 testnets."
    },
    {
        "JobDescriptionID": "JD_0036",
        "Category": "Engineering Manager",
        "Title": "Software Engineering Manager",
        "RequiredSkills": ["Project Management", "Agile", "Scrum", "SDLC", "Product Management"],
        "RequiredExperience": 8.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Manage software development teams, align tech roadmap with business metrics, and lead recruitment."
    },
    {
        "JobDescriptionID": "JD_0037",
        "Category": "Principal Engineer",
        "Title": "Principal Architect",
        "RequiredSkills": ["System Design", "AWS", "Docker", "Kubernetes", "Java", "Python"],
        "RequiredExperience": 10.0,
        "RequiredEducation": "Master's",
        "Description": "Define architecture standards across the company. Lead migration to microservices on Kubernetes and define technical roadmaps."
    },
    {
        "JobDescriptionID": "JD_0038",
        "Category": "Product Manager",
        "Title": "Technical Product Manager",
        "RequiredSkills": ["Product Management", "Agile", "JIRA", "Business Analysis", "Requirements Gathering"],
        "RequiredExperience": 5.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Define product vision, run prioritization sprints, draft PRDs, and manage JIRA tickets with developers."
    },
    {
        "JobDescriptionID": "JD_0039",
        "Category": "Technical Writer",
        "Title": "API & Technical Writer",
        "RequiredSkills": ["Technical Writing", "Documentation", "Markdown", "Git", "API Documentation"],
        "RequiredExperience": 3.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Create reference manuals, tutorials, and developer guides. Standardize Markdown formatting and document REST API surfaces."
    },
    {
        "JobDescriptionID": "JD_0040",
        "Category": "DevOps",
        "Title": "Site Reliability / DevOps Lead",
        "RequiredSkills": ["Docker", "Kubernetes", "AWS", "Terraform", "SRE", "CI/CD"],
        "RequiredExperience": 7.0,
        "RequiredEducation": "Bachelor's",
        "Description": "Lead a team of SRE and DevOps engineers managing multi-region cloud infrastructures on AWS."
    }
]

def generate_synthetic_jds(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for jd in SEED_JOB_DESCRIPTIONS:
            f.write(json.dumps(jd) + "\n")
    print(f"Generated {len(SEED_JOB_DESCRIPTIONS)} synthetic Job Descriptions in {output_path}")

if __name__ == "__main__":
    generate_synthetic_jds("c:/Users/bsais/OneDrive/Desktop/celabal/project/data/job_descriptions.jsonl")
