import re
import nltk
import spacy
from pdfminer.high_level import extract_text
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

nltk.download('punkt')

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load Hugging Face NER pipeline
try:
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
except RuntimeError as e:
    print("Failed to load the default NER model due to compatibility issues. Trying a different model.")
    # Load an alternative model that does not require tf-keras
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_name(text):
    ner_results = ner_pipeline(text)
    names = [result['word'] for result in ner_results if result['entity'] == 'B-PER']
    return " ".join(names) if names else None

def extract_contact_info(text):
    phone = re.findall(r'\b\d{10}\b', text)
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return phone, email

def extract_skills(text, known_skills):
    vectorizer = CountVectorizer().fit_transform([text] + known_skills)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    similarity_scores = cosine_matrix[0][1:]
    extracted_skills = [skill for skill, score in zip(known_skills, similarity_scores) if score > 0.1]
    return extracted_skills

def extract_education(text):
    education_keywords = ["Bachelor", "Master", "B.Sc", "M.Sc", "PhD", "Diploma"]
    sentences = nltk.sent_tokenize(text)
    education = []
    for sentence in sentences:
        for keyword in education_keywords:
            if keyword in sentence:
                education.append(sentence)
                break
    return education

def extract_experience(text):
    experience_keywords = ["experience", "worked", "developed", "managed", "led"]
    sentences = nltk.sent_tokenize(text)
    experience = []
    for sentence in sentences:
        for keyword in experience_keywords:
            if keyword in sentence.lower():
                experience.append(sentence)
                break
    return experience

def analyze_resume(pdf_path, known_skills, education_requirements):
    text = extract_text_from_pdf(pdf_path)
    name = extract_name(text)
    phone, email = extract_contact_info(text)
    skills = extract_skills(text, known_skills)
    education = extract_education(text)
    experience = extract_experience(text)

    education_matches = [edu for edu in education if any(req in edu for req in education_requirements)]
    
    return {
        "Name": name,
        "Phone": phone,
        "Email": email,
        "Skills": skills,
        "Education": education_matches,
        "Experience": experience
    }

def display_results(analysis):
    result_window = tk.Toplevel(root)
    result_window.title("Resume Analysis Result")

    data = [
        ["Name", analysis["Name"]],
        ["Phone", ", ".join(analysis["Phone"]) if analysis["Phone"] else "N/A"],
        ["Email", ", ".join(analysis["Email"]) if analysis["Email"] else "N/A"],
        ["Skills", ", ".join(analysis["Skills"]) if analysis["Skills"] else "N/A"],
        ["Education", "\n".join(analysis["Education"]) if analysis["Education"] else "N/A"],
        ["Experience", "\n".join(analysis["Experience"]) if analysis["Experience"] else "N/A"]
    ]

    table = tabulate(data, headers=["Field", "Details"], tablefmt="grid")

    text_result = tk.Text(result_window, wrap=tk.WORD, font=("Times New Roman", 12, "bold"))
    text_result.insert(tk.END, table)
    text_result.config(state=tk.DISABLED)
    text_result.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

def analyze():
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        messagebox.showerror("Error", "Please select a resume PDF file.")
        return

    known_skills = entry_skills.get().split(',')
    known_skills = [skill.strip() for skill in known_skills]

    education_requirements = entry_education.get().split(',')
    education_requirements = [req.strip() for req in education_requirements]

    analysis = analyze_resume(pdf_path, known_skills, education_requirements)
    display_results(analysis)

# Set up the main application window
root = tk.Tk()
root.title("Resume Analyzer")
root.geometry("800x500")
root.configure(bg='ivory')
root.resizable(False, False)

# Set up the UI elements
frame = tk.Frame(root, padx=10, pady=10, bg='ivory')
frame.pack(fill=tk.BOTH, expand=True)

# Heading
label_heading = tk.Label(frame, text="CV Analysis", font=("Times New Roman", 24, "bold"), bg='ivory')
label_heading.grid(row=0, column=0, columnspan=3, pady=10)

label_skills = tk.Label(frame, text="Required Skills (comma-separated):", font=("Times New Roman", 16, "bold"), bg='ivory')
label_skills.grid(row=1, column=0, sticky=tk.W, pady=5)

entry_skills = tk.Entry(frame, width=50)
entry_skills.grid(row=1, column=1, columnspan=2, pady=5)

label_education = tk.Label(frame, text="Education Requirements (comma-separated):", font=("Times New Roman", 16, "bold"), bg='ivory')
label_education.grid(row=2, column=0, sticky=tk.W, pady=5)

entry_education = tk.Entry(frame, width=50)
entry_education.grid(row=2, column=1, columnspan=2, pady=5)

button_analyze = tk.Button(frame, text="Analyze Resume", command=analyze, font=("Times New Roman", 16, "bold"))
button_analyze.grid(row=3, column=1, pady=10)

root.mainloop()
