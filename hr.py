import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# Add an image at the top
st.image("https://github.com/fortigateguru/hr/blob/main/ai-generated-8772394_640.png", caption="Best Match HR App", use_column_width=True)

# Title
st.title("Best Match HR App")

# Subheader and job description input
st.subheader("Enter job description")
job_description = st.text_area("Job Description", 
                               "Looking for a data scientist, junior level, skilled in LLM and Retrieval-Augmented Generation (RAG) techniques.")


# Step 1: Prepare the CVs (same as before)
cv_texts = [
    "Data scientist with experience in Python, machine learning, and data analysis.",
    "Experienced software developer skilled in Java, Python, and AWS cloud services.",
    "Graphic designer proficient in Adobe Photoshop, Illustrator, and digital art.",
    "Machine learning engineer with expertise in Python, deep learning, and TensorFlow.",
    "Project manager with experience in agile methodologies, project planning, and Jira.",
    "System administrator experienced with Linux, networking, and cloud services.",
    "Frontend developer skilled in HTML, CSS, JavaScript, and React.",
    "Backend developer with expertise in Node.js, Express, and database management.",
    "Cybersecurity analyst with knowledge of network security, firewalls, and ethical hacking.",
    "Business analyst with experience in data modeling, SQL, and business intelligence tools."
]

# Step 2: Load a pre-trained BERT model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Step 3: Generate embeddings for the CVs
cv_embeddings = model.encode(cv_texts, convert_to_tensor=True)

# Step 4: Create the Streamlit user interface
st.title("Best Match HR App")

st.subheader("Enter job description")
job_description = st.text_area("Job Description",
                               "Looking for a data scientist, junior level, skilled in LLM and Retrieval-Augmented Generation (RAG) techniques.")

# Step 5: Option to trigger the match-making process
if st.button("Find Best Match"):
    # Step 6: Encode the job description
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    # Step 7: Find the similarity between the job description and the CVs
    similarity_scores = util.pytorch_cos_sim(job_embedding, cv_embeddings)

    # Step 8: Sort CVs by similarity
    sorted_indices = torch.argsort(similarity_scores, descending=True).tolist()[0]

    # Step 9: Display the top matches
    st.subheader("Top Matches")

    for idx in sorted_indices:
        st.write(f"CV {idx + 1}: {cv_texts[idx]}")
        st.write(f"Similarity Score: {similarity_scores[0][idx].item():.4f}")
        st.write("---")
