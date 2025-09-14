import os
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from bson import ObjectId

app = Flask(__name__)

# -------------------------------
# 1. MongoDB Connection
# -------------------------------
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://root:root@laxman.xmhzqpt.mongodb.net/PmInternshipScheme?retryWrites=true&w=majority&appName=laxman"
)
client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client["PmInternshipScheme"]
allocations_col = db["allocations"]

# -------------------------------
# 2. Load internships dataset once
# -------------------------------
internships = pd.read_csv("internships.csv")
internships["requirements"] = (
    internships["RequiredSkills"].astype(str) + " " +
    internships["QualificationRequired"].astype(str) + " " +
    internships["WorkLocation"].astype(str)
)

vectorizer = CountVectorizer()
vectorizer.fit(internships["requirements"].tolist())
internship_vecs = torch.tensor(
    vectorizer.transform(internships["requirements"]).toarray(),
    dtype=torch.float32
)

cos = nn.CosineSimilarity(dim=1)

# -------------------------------
# 3. Define Aspirational & Rural Districts
# -------------------------------
aspirational_districts = {
    ("Mewat", "Haryana"),
    ("Nuh", "Haryana"),
    ("Kalahandi", "Odisha"),
    ("Dhar", "Madhya Pradesh"),
    ("Koraput", "Odisha"),
    ("Gaya", "Bihar"),
    ("Dantewada", "Chhattisgarh"),
    ("Barmer", "Rajasthan"),
    ("Nandurbar", "Maharashtra"),
    ("Rajgarh", "Madhya Pradesh")
}

rural_districts = {
    ("Nuh", "Haryana"),
    ("Kalahandi", "Odisha"),
    ("Koraput", "Odisha"),
    ("Dantewada", "Chhattisgarh"),
    ("Nandurbar", "Maharashtra")
}

# -------------------------------
# 4. Utils
# -------------------------------
def clean_for_json(docs):
    """Convert ObjectId to string recursively for JSON serialization"""
    if isinstance(docs, list):
        return [clean_for_json(d) for d in docs]
    if isinstance(docs, dict):
        return {k: clean_for_json(v) for k, v in docs.items()}
    if isinstance(docs, ObjectId):
        return str(docs)
    return docs

# -------------------------------
# 5. Routes
# -------------------------------
@app.route("/test", methods=["GET"])
def test_route():
    return "Working"


@app.route("/match", methods=["POST"])
def match_opportunities():
    data = request.json
    applicant_id = data.get("applicant_id")

    if not applicant_id:
        return jsonify({"error": "applicant_id is required"}), 400

    # -------------------------------
    # Step 1: Check if allocations already exist for applicant
    # -------------------------------
    existing_allocations = list(allocations_col.find({"applicant_id": applicant_id}))
    if existing_allocations:
        return jsonify(clean_for_json(existing_allocations))  # return stored data directly

    # -------------------------------
    # Step 2: Build text profile
    # -------------------------------
    skills = data.get("skills", [])
    skills_str = " ".join(skills) if isinstance(skills, list) else str(skills)

    profile = f"{skills_str} {data.get('qualifications','')} {data.get('location_preferences','')} {data.get('native_location','')} {data.get('social_category','')}"
    student_vec = torch.tensor(
        vectorizer.transform([profile]).toarray(),
        dtype=torch.float32
    )

    similarities = cos(student_vec, internship_vecs).detach().numpy().flatten()

    # -------------------------------
    # Extract district, state safely
    # -------------------------------
    native_parts = [p.strip() for p in data.get("native_location", "").split(",")]
    district, state = (native_parts[0], native_parts[1]) if len(native_parts) == 2 else ("", "")

    is_aspirational = (district, state) in aspirational_districts
    is_rural = (district, state) in rural_districts

    results = []
    for i, score in enumerate(similarities):
        final_score = score

        # -------------------------------
        # Apply Affirmative Action Bonuses
        # -------------------------------
        sc = data.get("social_category", "")
        if sc in ["SC", "ST"]:
            final_score += 0.15
        elif sc in ["OBC", "BC", "SBC", "EWS"]:
            final_score += 0.10

        if is_aspirational:
            final_score += 0.10
        if is_rural:
            final_score += 0.05

        # -------------------------------
        # Apply Participation Priority
        # -------------------------------
        status = data.get("participation_status", "")
        if status == "Rejected":
            final_score += 0.15
        elif status == "New":
            final_score += 0.10
        elif status == "Benefitted":
            final_score += 0.00

        # Clamp score
        final_score = min(final_score, 1.0)

        if final_score >= 0.6:
            results.append({
                "applicant_id": applicant_id,
                "internship": internships.iloc[i]["Role"],
                "company": internships.iloc[i]["Company"],
                "raw_similarity": round(float(score), 3),
                "final_score": round(float(final_score), 3),
                "is_aspirational": is_aspirational,
                "is_rural": is_rural
            })

    # Sort by final score
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # -------------------------------
    # Step 3: Save new allocations
    # -------------------------------
    if results:
        allocations_col.insert_many(results)

    return jsonify(clean_for_json(results))


if __name__ == "__main__":
    app.run(port=5001, debug=True)
