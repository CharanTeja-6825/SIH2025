import docx2txt
import pdfplumber

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

txtex = extract_text_from_pdf("../../content/data/data/ACCOUNTANT/10554236.pdf")
lines = [line.strip() for line in txtex.split("\n") if line.strip()]

# Predefined headers you expect in resumes
HEADERS = {"Skills", "Highlights", "Accomplishments", "Experience", "Education"}

results = {}
current_section = None

for line in lines:
    if line in HEADERS:  # Found a new section
        current_section = line
        results[current_section] = []
    elif current_section:
        results[current_section].append(line)

# Convert lists into single text blocks
for key in results:
    results[key] = " ".join(results[key])
    print(key, " ", results[key])
