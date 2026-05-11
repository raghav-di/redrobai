import math

# Resume data
resumes = [
    {"name": "Sneha Singh", "raw_skills": "python, machine learning, sql, pandas, numpy"},
    {"name": "Meera Kapoor", "raw_skills": "java, deep learning, html/css, javascript, react"},
    {"name": "Karan Malhotra", "raw_skills": "python, machine learning, sql, pandas, numpy"},
    {"name": "Arjun Sharma", "raw_skills": "python, ml, data-viz, matplotlib, power bi"}
]

SKILL_ALIASES = {
    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",
    # Web — Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",
    # Web — Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",
    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",
    # Mobile
    "android": "android",
    "firebase": "firebase",
    # CS Fundamentals
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}


def normalizeskills(rawskills):
    skills = []
    if not isinstance(rawskills, str):
        return skills

    for skill in rawskills.split(","):
        canonical = SKILL_ALIASES.get(skill.strip().lower())
        if canonical:
            skills.append(canonical)

    return skills


def deduplicateskills(normalizedskills):
    return sorted(set(normalizedskills))


def construct_vocabulary(resumes):
    vocabulary = set()
    for resume in resumes:
        vocabulary.update(resume.get("normalized_skills", []))
    return sorted(vocabulary)


def calculatetfidf(resumes, vocabulary):
    df = {
        skill: sum(1 for resume in resumes if skill in resume.get("normalized_skills", []))
        for skill in vocabulary
    }

    tfidfvectors = []
    for resume in resumes:
        normalized_skills = resume.get("normalized_skills", [])
        tfidfvector = []

        for skill in vocabulary:
            tf = 1 / len(normalized_skills) if normalized_skills and skill in normalized_skills else 0
            idf = math.log(len(resumes) / df[skill]) if df[skill] > 0 else 0
            tfidfvector.append(tf * idf)

        tfidfvectors.append(tfidfvector)

    return tfidfvectors


def constructjdvector(jd, vocabulary):
    if not isinstance(jd, dict) or "required_skills" not in jd or "preferred_skills" not in jd:
        raise ValueError("Invalid job description input")

    if not isinstance(vocabulary, list) or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("Invalid vocabulary input")

    jd_vector = [0] * len(vocabulary)
    required_skills = jd["required_skills"]
    preferred_skills = jd["preferred_skills"]

    for skill in required_skills + preferred_skills:
        canonical = SKILL_ALIASES.get(skill.strip().lower(), skill.strip().lower())
        if canonical in vocabulary:
            index = vocabulary.index(canonical)
            jd_vector[index] = 1

    return jd_vector


def calculatecosinesimilarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vector1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vector2))

    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0.0


def calculateresumejdsimilarity(resumetfidf, jd_vector):
    return calculatecosinesimilarity(resumetfidf, jd_vector)


job_descriptions = [
    {
        "name": "JD-1",
        "required_skills": ["python", "machine learning", "sql", "pandas", "numpy"],
        "preferred_skills": ["nlp", "bert", "feature engineering", "statistics"],
    },
    {
        "name": "JD-2",
        "required_skills": ["java", "spring boot", "mysql", "microservices", "docker", "kubernetes"],
        "preferred_skills": ["rest api", "ci/cd", "redis"],
    },
    {
        "name": "JD-3",
        "required_skills": ["javascript", "react", "vue", "typescript", "rest api", "html/css"],
        "preferred_skills": ["node.js", "graphql", "redux", "jest", "aws"],
    },
]


def build_normalized_resumes(resumes_list):
    normalized_resumes = []
    for index, resume in enumerate(resumes_list, start=1):
        normalized = normalizeskills(resume.get("raw_skills", ""))
        normalized_resumes.append(
            {
                "name": resume.get("name", f"Resume {index}"),
                "raw_skills": resume.get("raw_skills", ""),
                "normalized_skills": deduplicateskills(normalized),
            }
        )
    return normalized_resumes


def main():
    normalized_resumes = build_normalized_resumes(resumes)
    vocabulary = construct_vocabulary(normalized_resumes)
    resume_tfidf_vectors = calculatetfidf(normalized_resumes, vocabulary)

    for resume, vector in zip(normalized_resumes, resume_tfidf_vectors):
        resume["tfidf"] = vector

    jd_vectors = []
    for jd in job_descriptions:
        jd_vectors.append(constructjdvector(jd, vocabulary))

    similarities = []
    for resume in normalized_resumes:
        for jd, jd_vector in zip(job_descriptions, jd_vectors):
            similarities.append(
                {
                    "resume": resume["name"],
                    "jd": jd["name"],
                    "similarity": round(calculateresumejdsimilarity(resume["tfidf"], jd_vector), 2),
                }
            )

    for jd in job_descriptions:
        jd_name = jd["name"]
        top_matches = sorted(
            [sim for sim in similarities if sim["jd"] == jd_name],
            key=lambda item: item["similarity"],
            reverse=True,
        )
        print(f"{jd_name} Result: Top 3 Candidates with Matching Scores.")
        candidates = [f"{match['resume']}({match['similarity']})" for match in top_matches[:3]]
        print(", ".join(candidates))
        print("*")