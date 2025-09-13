from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
import os

app = Flask(__name__)

# -------------------------------
# 1. MongoDB Setup
# -------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://root:root@laxman.xmhzqpt.mongodb.net/PmInternshipScheme?retryWrites=true&w=majority&appName=laxman")
client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client["PmInternshipScheme"]          # database name
allocations_col = db["allocations"]   # collection name

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
# 4. Routes
# -------------------------------
@app.route("/test", methods=["GET"])
def test_route():
    return "Working"


@app.route("/match", methods=["POST"])
def match_opportunities():
    data = request.json  # student details

    # -------------------------------
    # Build text profile
    # -------------------------------
    skills_str = " ".join(data["skills"]) if isinstance(data["skills"], list) else str(data["skills"])
    profile = f"{skills_str} {data['qualifications']} {data['location_preferences']} {data['native_location']} {data['social_category']}"

    student_vec = torch.tensor(
        vectorizer.transform([profile]).toarray(),
        dtype=torch.float32
    )
    similarities = cos(student_vec, internship_vecs).detach().numpy().flatten()

    # -------------------------------
    # Extract district, state
    # -------------------------------
    native_parts = [p.strip() for p in data["native_location"].split(",")]
    district, state = (native_parts[0], native_parts[1]) if len(native_parts) == 2 else ("", "")

    is_aspirational = (district, state) in aspirational_districts
    is_rural = (district, state) in rural_districts

    results = []
    for i, score in enumerate(similarities):
        final_score = score

        # Affirmative action bonuses
        if data["social_category"] in ["SC", "ST"]:
            final_score += 0.15
        elif data["social_category"] in ["OBC", "BC", "SBC", "EWS"]:
            final_score += 0.10

        if is_aspirational:
            final_score += 0.10
        if is_rural:
            final_score += 0.05

        # Participation priority
        status = data["participation_status"]
        if status == "Rejected":
            final_score += 0.15
        elif status == "New":
            final_score += 0.10
        elif status == "Benefitted":
            final_score += 0.00

        final_score = min(final_score, 1.0)  # clamp

        if final_score >= 0.75:
            results.append({
                "student_id": data.get("student_id", None),   # optional if passed
                "internship": internships.iloc[i]["Role"],
                "company": internships.iloc[i]["Company"],
                "raw_similarity": round(float(score), 3),
                "final_score": round(float(final_score), 3),
                "is_aspirational": is_aspirational,
                "is_rural": is_rural,
                "social_category": data["social_category"],
                "participation_status": data["participation_status"]
            })

    # Sort by final score
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # -------------------------------
    # Store in MongoDB
    # -------------------------------
    if results:
        allocations_col.insert_many(results)

    return jsonify({"status": "stored", "count": len(results)})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
