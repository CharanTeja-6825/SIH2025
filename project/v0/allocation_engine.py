import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# 1. Load datasets
# -------------------------------
students = pd.read_csv("students_diverse_with_asp.csv")
internships = pd.read_csv("internships.csv")

# Combine fields into single text descriptions
students["profile"] = (
    students["Skills"].astype(str) + " " +
    students["Qualification"].astype(str) + " " +
    students["PreferredLocation"].astype(str) + " " +
    students["NativeLocation"].astype(str) + " " +
    students["SocialCategory"].astype(str)
)

internships["requirements"] = (
        internships["RequiredSkills"].astype(str) + " "
        + internships["QualificationRequired"].astype(str) + " "
        + internships["WorkLocation"].astype(str)
)

# -------------------------------
# 2. Convert text to embeddings
# -------------------------------
vectorizer = CountVectorizer()
vectorizer.fit(students["profile"].tolist() + internships["requirements"].tolist())

student_vecs = torch.tensor(vectorizer.transform(students["profile"]).toarray(), dtype=torch.float32)
internship_vecs = torch.tensor(vectorizer.transform(internships["requirements"]).toarray(), dtype=torch.float32)

# -------------------------------
# 3. Similarity + Fairness Boost
# -------------------------------
cos = nn.CosineSimilarity(dim=1)

allocations = []
for i, student in students.iterrows():
    student_vector = student_vecs[i].unsqueeze(0)
    similarities = cos(student_vector, internship_vecs)

    # Get best match
    best_idx = torch.argmax(similarities).item()
    base_score = similarities[best_idx].item()

    # Fairness Boosts
    bonus = 0.0

    # Social category bonus
    if student["SocialCategory"] in ["SC", "ST"]:
        bonus += 0.15
    elif student["SocialCategory"] in ["EWS", "OBC"]:
        bonus += 0.1

    # Aspirational district bonus
    if student["IsAspirationalDistrict"] == 1:
        bonus += 0.1

    # Final Score
    final_score = min(base_score + bonus, 1.0)  # cap at 1.0

    allocations.append({
        "Student": student["StudentID"],
        "BestInternship": internships.iloc[best_idx]["Role"],
        "Company": internships.iloc[best_idx]["Company"],
        "BaseScore": round(base_score, 3),
        "FairnessBonus": round(bonus, 3),
        "FinalScore": round(final_score, 3)
    })

# -------------------------------
# 4. Save results
# -------------------------------
alloc_df = pd.DataFrame(allocations)
print(alloc_df.head(10))
alloc_df.to_csv("allocations_pytorch_fair.csv", index=False)
