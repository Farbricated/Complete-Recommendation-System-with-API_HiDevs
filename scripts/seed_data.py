"""
Seed the database with rich test data.
Creates 10+ users, 20+ content items, skills, and realistic interaction patterns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from data.database import init_db, get_db
from data.repositories import UserRepository, ContentRepository, InteractionRepository

random.seed(42)

# ---------------------------------------------------------------------------
# Data definitions
# ---------------------------------------------------------------------------

USERS = [
    {"username": "alice_dev",     "email": "alice@example.com",   "skill_level": "advanced",     "preferences": ["python", "machine-learning", "data-science"]},
    {"username": "bob_student",   "email": "bob@example.com",     "skill_level": "beginner",     "preferences": ["web-dev", "javascript", "html"]},
    {"username": "carol_analyst", "email": "carol@example.com",   "skill_level": "intermediate", "preferences": ["data-science", "sql", "visualization"]},
    {"username": "david_ml",      "email": "david@example.com",   "skill_level": "advanced",     "preferences": ["machine-learning", "deep-learning", "python"]},
    {"username": "eve_designer",  "email": "eve@example.com",     "skill_level": "intermediate", "preferences": ["web-dev", "css", "ux"]},
    {"username": "frank_backend", "email": "frank@example.com",   "skill_level": "advanced",     "preferences": ["python", "databases", "cloud"]},
    {"username": "grace_new",     "email": "grace@example.com",   "skill_level": "beginner",     "preferences": ["python", "data-science"]},
    {"username": "henry_cloud",   "email": "henry@example.com",   "skill_level": "intermediate", "preferences": ["cloud", "devops", "databases"]},
    {"username": "iris_full",     "email": "iris@example.com",    "skill_level": "advanced",     "preferences": ["web-dev", "python", "databases"]},
    {"username": "jack_beginner", "email": "jack@example.com",    "skill_level": "beginner",     "preferences": ["javascript", "web-dev"]},
    {"username": "karen_ds",      "email": "karen@example.com",   "skill_level": "intermediate", "preferences": ["data-science", "machine-learning", "python"]},
    {"username": "leo_devops",    "email": "leo@example.com",     "skill_level": "advanced",     "preferences": ["devops", "cloud", "databases"]},
]

CONTENT = [
    # Python courses
    {"content_id": "c01", "title": "Python for Beginners",         "description": "Learn Python from scratch", "category": "python", "difficulty": "beginner",     "tags": ["python","programming","basics"],       "author": "Dr. Smith",  "duration_min": 120, "rating": 4.7},
    {"content_id": "c02", "title": "Advanced Python Patterns",     "description": "Design patterns in Python", "category": "python", "difficulty": "advanced",     "tags": ["python","patterns","oop"],             "author": "Jane Doe",   "duration_min": 90,  "rating": 4.5},
    {"content_id": "c03", "title": "Python Data Structures",       "description": "Algorithms & data structures", "category": "python", "difficulty": "intermediate", "tags": ["python","algorithms","dsa"],        "author": "Prof. Lee",  "duration_min": 150, "rating": 4.6},
    # Machine Learning
    {"content_id": "c04", "title": "Intro to Machine Learning",    "description": "ML fundamentals",          "category": "machine-learning", "difficulty": "beginner",     "tags": ["ml","sklearn","basics"],           "author": "Andrew Ng",  "duration_min": 180, "rating": 4.9},
    {"content_id": "c05", "title": "Deep Learning with PyTorch",   "description": "Neural networks deep dive", "category": "deep-learning",    "difficulty": "advanced",     "tags": ["pytorch","neural-nets","gpu"],     "author": "Yann L",     "duration_min": 240, "rating": 4.8},
    {"content_id": "c06", "title": "NLP with Transformers",        "description": "Hugging Face ecosystem",    "category": "machine-learning", "difficulty": "advanced",     "tags": ["nlp","transformers","bert"],       "author": "Thomas W",   "duration_min": 200, "rating": 4.7},
    {"content_id": "c07", "title": "ML Model Deployment",          "description": "Production ML systems",     "category": "machine-learning", "difficulty": "intermediate", "tags": ["mlops","deployment","docker"],     "author": "Sarah J",    "duration_min": 120, "rating": 4.4},
    # Data Science
    {"content_id": "c08", "title": "Data Analysis with Pandas",    "description": "Pandas mastery",           "category": "data-science", "difficulty": "intermediate", "tags": ["pandas","data-analysis","python"],  "author": "Wes M",      "duration_min": 100, "rating": 4.6},
    {"content_id": "c09", "title": "Data Visualization",           "description": "Matplotlib & Seaborn",     "category": "visualization",    "difficulty": "beginner",     "tags": ["matplotlib","seaborn","charts"],   "author": "Jake V",     "duration_min": 80,  "rating": 4.3},
    {"content_id": "c10", "title": "SQL for Data Scientists",      "description": "SQL analytics queries",     "category": "sql",      "difficulty": "intermediate", "tags": ["sql","analytics","databases"],      "author": "Mode A",     "duration_min": 110, "rating": 4.5},
    # Web Development
    {"content_id": "c11", "title": "Modern JavaScript ES6+",       "description": "JS fundamentals updated",  "category": "javascript", "difficulty": "beginner",     "tags": ["javascript","es6","web"],           "author": "Kyle S",     "duration_min": 140, "rating": 4.7},
    {"content_id": "c12", "title": "React from Zero to Hero",      "description": "Complete React course",     "category": "web-dev",  "difficulty": "intermediate", "tags": ["react","javascript","frontend"],     "author": "Maximilian", "duration_min": 300, "rating": 4.8},
    {"content_id": "c13", "title": "CSS Grid & Flexbox",           "description": "Modern CSS layouts",        "category": "css",      "difficulty": "beginner",     "tags": ["css","layout","web"],               "author": "Kevin P",    "duration_min": 60,  "rating": 4.4},
    {"content_id": "c14", "title": "Node.js REST APIs",            "description": "Build REST APIs with Node", "category": "web-dev",  "difficulty": "intermediate", "tags": ["nodejs","api","backend"],           "author": "Brad T",     "duration_min": 160, "rating": 4.5},
    # Databases & Cloud
    {"content_id": "c15", "title": "PostgreSQL Mastery",           "description": "Advanced PostgreSQL",       "category": "databases","difficulty": "advanced",     "tags": ["postgresql","sql","performance"],   "author": "Hamid M",    "duration_min": 130, "rating": 4.6},
    {"content_id": "c16", "title": "AWS for Beginners",            "description": "Cloud fundamentals",        "category": "cloud",    "difficulty": "beginner",     "tags": ["aws","cloud","s3","ec2"],           "author": "Stephane M", "duration_min": 200, "rating": 4.7},
    {"content_id": "c17", "title": "Docker & Kubernetes",          "description": "Container orchestration",  "category": "devops",   "difficulty": "intermediate", "tags": ["docker","kubernetes","containers"],  "author": "Mumshad M",  "duration_min": 250, "rating": 4.8},
    {"content_id": "c18", "title": "Redis Caching Strategies",     "description": "High-performance caching",  "category": "databases","difficulty": "intermediate", "tags": ["redis","caching","performance"],    "author": "Redis Labs", "duration_min": 75,  "rating": 4.4},
    # Stats & Math
    {"content_id": "c19", "title": "Statistics for Data Science",  "description": "Probability and stats",     "category": "data-science", "difficulty": "intermediate", "tags": ["statistics","probability","math"],"author": "Joe B",      "duration_min": 180, "rating": 4.5},
    {"content_id": "c20", "title": "Linear Algebra for ML",        "description": "Math behind ML",            "category": "machine-learning", "difficulty": "intermediate", "tags": ["math","linear-algebra","ml"],  "author": "3Blue1B",    "duration_min": 120, "rating": 4.9},
    # Extra content for richness
    {"content_id": "c21", "title": "FastAPI in Practice",          "description": "Build APIs with FastAPI",   "category": "python",   "difficulty": "intermediate", "tags": ["fastapi","python","api"],           "author": "Sebastián",  "duration_min": 90,  "rating": 4.7},
    {"content_id": "c22", "title": "UX Design Principles",         "description": "User experience design",    "category": "ux",       "difficulty": "beginner",     "tags": ["ux","design","wireframing"],       "author": "Don Norman", "duration_min": 100, "rating": 4.3},
]

SKILLS = [
    {"skill_id": "sk01", "name": "Python Programming",   "category": "programming"},
    {"skill_id": "sk02", "name": "Machine Learning",     "category": "ai"},
    {"skill_id": "sk03", "name": "Deep Learning",        "category": "ai"},
    {"skill_id": "sk04", "name": "Data Analysis",        "category": "data"},
    {"skill_id": "sk05", "name": "SQL",                  "category": "data"},
    {"skill_id": "sk06", "name": "JavaScript",           "category": "programming"},
    {"skill_id": "sk07", "name": "React",                "category": "frontend"},
    {"skill_id": "sk08", "name": "AWS",                  "category": "cloud"},
    {"skill_id": "sk09", "name": "Docker",               "category": "devops"},
    {"skill_id": "sk10", "name": "Data Visualization",  "category": "data"},
]

# content_id → [skill_ids]
CONTENT_SKILLS = {
    "c01": ["sk01"], "c02": ["sk01"], "c03": ["sk01"],
    "c04": ["sk02", "sk01"], "c05": ["sk03", "sk01"], "c06": ["sk02", "sk03"],
    "c07": ["sk02"], "c08": ["sk04", "sk01"], "c09": ["sk10"],
    "c10": ["sk05"], "c11": ["sk06"], "c12": ["sk06", "sk07"],
    "c13": ["sk06"], "c14": ["sk06"], "c15": ["sk05"],
    "c16": ["sk08"], "c17": ["sk09"], "c18": ["sk05"],
    "c19": ["sk04"], "c20": ["sk02"], "c21": ["sk01"], "c22": [],
}

# Realistic interaction patterns per user
INTERACTION_PATTERNS = {
    "alice_dev":     [("c02","complete",5),("c03","complete",4),("c04","view",None),("c05","bookmark",None),("c21","complete",5)],
    "bob_student":   [("c01","complete",5),("c11","view",None),("c13","click",None),("c09","view",None)],
    "carol_analyst": [("c08","complete",5),("c10","complete",4),("c09","complete",5),("c19","view",None),("c04","click",None)],
    "david_ml":      [("c05","complete",5),("c06","complete",5),("c20","complete",5),("c04","bookmark",None),("c07","view",None)],
    "eve_designer":  [("c13","complete",5),("c12","view",None),("c11","complete",4),("c22","click",None)],
    "frank_backend": [("c15","complete",5),("c17","complete",5),("c18","view",None),("c16","complete",4),("c21","bookmark",None)],
    "henry_cloud":   [("c16","complete",5),("c17","complete",4),("c15","view",None),("c10","click",None)],
    "iris_full":     [("c12","complete",5),("c14","complete",5),("c15","view",None),("c21","click",None),("c01","view",None)],
    "jack_beginner": [("c11","view",None),("c01","click",None),("c13","view",None)],
    "karen_ds":      [("c08","complete",5),("c19","complete",4),("c04","complete",5),("c20","view",None),("c09","click",None)],
    "leo_devops":    [("c17","complete",5),("c16","complete",5),("c15","bookmark",None),("c18","complete",4)],
}


def seed():
    """Run the full seeding process."""
    print("🌱 Seeding database...")
    init_db()

    user_repo    = UserRepository()
    content_repo = ContentRepository()
    interaction_repo = InteractionRepository()

    # --- Users ---
    user_ids = {}
    for u in USERS:
        existing = user_repo.get_by_username(u["username"])
        if existing:
            user_ids[u["username"]] = existing["user_id"]
            continue
        created = user_repo.create(**u)
        user_ids[u["username"]] = created["user_id"]
    print(f"  ✓ {len(user_ids)} users seeded")

    # --- Skills ---
    with get_db() as conn:
        for sk in SKILLS:
            conn.execute(
                "INSERT OR IGNORE INTO skills (skill_id, name, category) VALUES (?, ?, ?)",
                (sk["skill_id"], sk["name"], sk["category"])
            )
    print(f"  ✓ {len(SKILLS)} skills seeded")

    # --- Content ---
    for c in CONTENT:
        existing = content_repo.get(c["content_id"])
        if existing:
            continue
        content_repo.create(**c)
    print(f"  ✓ {len(CONTENT)} content items seeded")

    # --- Content-skill mappings ---
    for cid, skill_ids in CONTENT_SKILLS.items():
        for sid in skill_ids:
            content_repo.add_skill(cid, sid)
    print(f"  ✓ Content-skill mappings seeded")

    # --- Interactions ---
    event_map = {"complete": "complete", "view": "view", "click": "click", "bookmark": "bookmark"}
    total_interactions = 0
    for username, events in INTERACTION_PATTERNS.items():
        uid = user_ids.get(username)
        if not uid:
            continue
        for cid, etype, rating in events:
            interaction_repo.record(
                user_id=uid,
                content_id=cid,
                event_type=etype,
                rating=float(rating) if rating else None,
            )
            total_interactions += 1
    print(f"  ✓ {total_interactions} interactions seeded")

    print("\n✅ Database seeded successfully!")
    print(f"   Users: {len(USERS)}, Content: {len(CONTENT)}, Skills: {len(SKILLS)}")
    print(f"   Interactions: {total_interactions}")
    return user_ids


if __name__ == "__main__":
    user_ids = seed()
    print("\nUser IDs:")
    for username, uid in user_ids.items():
        print(f"  {username}: {uid}")