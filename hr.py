
import streamlit as st
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer, util
import torch

# Add an image from GitHub (as per your provided link)
st.image("https://github.com/fortigateguru/hr/blob/main/puzzle-3155663_1280.png?raw=true", 
         caption="Best Match HR App", use_column_width=True)

# Function to parse a PDF file
def parse_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to parse a DOCX file
def parse_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

# Function to handle file upload and parse based on type
def parse_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return parse_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return parse_docx(uploaded_file)
    else:
        st.warning("Unsupported file type. Please upload a PDF or DOCX file.")
        return None

# App title and description
st.title("Best Match HR App with Real-Time CV Parsing")

# Job description input
st.subheader("הכנס תיאור משרה")
job_description = st.text_area("Job Description", "Looking for a data scientist with experience in Python, machine learning, and deep learning.")

# Upload CV files
st.subheader("העלה קורות חיים (PDF or DOCX)")
uploaded_files = st.file_uploader("Upload CV files", type=["pdf", "docx"], accept_multiple_files=True)

# Load the BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Process the uploaded files
cv_texts = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        parsed_text = parse_uploaded_file(uploaded_file)
        if parsed_text:
            cv_texts.append(parsed_text)
            st.write(f"Extracted text from {uploaded_file.name}:", parsed_text[:500])  # Displaying only first 500 characters

# Matching process
if cv_texts and st.button("מצא את המועמד הכי מתאים"):
    # Encode the job description
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    
    # Generate embeddings for the CVs
    cv_embeddings = model.encode(cv_texts, convert_to_tensor=True)
    
    # Find similarity between job description and CVs
    similarity_scores = util.pytorch_cos_sim(job_embedding, cv_embeddings)
    
    # Sort CVs by similarity
    sorted_indices = torch.argsort(similarity_scores, descending=True).tolist()[0]
    
    # Display the top matches
    st.subheader("Top Matches")
    for idx in sorted_indices:
        st.write(f"CV {idx+1} (from {uploaded_files[idx].name}):")
        st.write(f"Similarity Score: {similarity_scores[0][idx].item():.4f}")
        st.write("---")

