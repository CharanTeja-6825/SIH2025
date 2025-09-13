import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# 1. Load internships dataset
# -------------------------------
internships = pd.read_csv("internships.csv")
internships["requirements"] = (
        internships["RequiredSkills"].astype(str) + " " +
        internships["QualificationRequired"].astype(str) + " " +
        internships["WorkLocation"].astype(str)
)

# Fit vectorizer only once on internships
vectorizer = CountVectorizer()
vectorizer.fit(internships["requirements"].tolist())
internship_vecs = torch.tensor(
    vectorizer.transform(internships["requirements"]).toarray(),
    dtype=torch.float32
)

cos = nn.CosineSimilarity(dim=1)


# -------------------------------
# 2. Function to compute matching for a single student
# -------------------------------
def match_student(student_record, threshold=0.6):
    """
    student_record: dict with keys:
        'Skills', 'Qualification', 'PreferredLocation',
        'NativeLocation', 'SocialCategory', 'IsAspirationalDistrict'
    """
    # Build profile string
    profile = f"{student_record['skills']} {student_record['qualifications']} " \
              f"{student_record['location_preferences']} {student_record['native_location']} " \
              f"{student_record['social_category']}"

    student_vec = torch.tensor(
        vectorizer.transform([profile]).toarray(),
        dtype=torch.float32
    )

    # Compute similarity with all internships
    similarities = cos(student_vec, internship_vecs).detach().numpy().flatten()

    results = []
    for i, score in enumerate(similarities):
        if score >= threshold:
            # Apply affirmative action bonus
            bonus = 0.0
            if student_record['social_category'] in ['SC', 'ST']:
                bonus += 0.15
            elif student_record['social_category'] in ['EWS', 'OBC']:
                bonus += 0.1
            if student_record['is_aspirational'] == 1:
                bonus += 0.1

            final_score = min(score + bonus, 1.0)
            results.append({
                'internship': internships.iloc[i]['Role'],
                'company': internships.iloc[i]['Company'],
                'base_score': round(score, 3),
                'bonus': round(bonus, 3),
                'final_score': round(final_score, 3)
            })

    # Sort descending by final score
    results = sorted(results, key=lambda x: x['final_score'], reverse=True)
    return results


# -------------------------------
# 3. Example usage
# -------------------------------
# Load a single student record (first row of your CSV)
students = pd.read_csv("students_diverse_districts.csv")
student = students.iloc[0].to_dict()

matched_opportunities = match_student(student, threshold=0.6)

# Print results
for idx, op in enumerate(matched_opportunities, 1):
    print(
        f"{idx}. {op['internship']} @ {op['company']}, Score: {op['final_score']}, Base: {op['base_score']}, Bonus: {op['bonus']}")
