import pandas as pd
import random

# List of states and districts (mix of aspirational + non-aspirational)
aspirational_districts = [
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
]

normal_districts = [
    ("Hyderabad", "Telangana"),
    ("Bengaluru", "Karnataka"),
    ("Chennai", "Tamil Nadu"),
    ("Mumbai", "Maharashtra"),
    ("Visakhapatnam", "Andhra Pradesh"),
    ("Pune", "Maharashtra"),
    ("Mysuru", "Karnataka"),
    ("Jaipur", "Rajasthan"),
    ("Patna", "Bihar"),
    ("Bhopal", "Madhya Pradesh"),
    ("Lucknow", "Uttar Pradesh"),
    ("Kochi", "Kerala"),
    ("Coimbatore", "Tamil Nadu"),
    ("Nagpur", "Maharashtra"),
    ("Varanasi", "Uttar Pradesh"),
]

all_districts = aspirational_districts + normal_districts

skills = ["Python", "Java", "C++", "Machine Learning", "Deep Learning", "SQL", "Spring Boot", "React", "Angular",
          "Data Analysis"]
qualifications = ["B.Tech", "M.Tech", "MBA", "B.Sc", "MCA"]
social_categories = ["OC", "BC", "SC", "ST", "SBC"]

students_data = []

for i in range(100):
    # Randomly pick district/state
    district, state = random.choice(all_districts)
    native_location = f"{district}, {state}"

    # Check if it's aspirational
    is_asp = 1 if (district, state) in aspirational_districts else 0

    student = {
        "student_id": f"S{i + 1:03d}",
        "skills": random.choice(skills),
        "qualifications": random.choice(qualifications),
        "location_preferences": random.choice([state, "Remote", "Any"]),
        "native_location": native_location,
        "social_category": random.choice(social_categories),
        "is_aspirational": is_asp
    }
    students_data.append(student)

# Save to CSV
students_diverse_districts = pd.DataFrame(students_data)
students_diverse_districts.to_csv("./students_diverse_districts.csv", index=False)

students_diverse_districts.head(10)
