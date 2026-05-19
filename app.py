import numpy as np
import pypdf
from sentence_transformers import SentenceTransformer
import anthropic
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def process_pdf(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    chunks = split_text(full_text)
    model = load_embedding_model()
    embeddings = model.encode(chunks)
    return chunks, embeddings

def split_text(text, chunks_size = 500, overlap =50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunks_size])
        start += chunks_size - overlap
    return chunks

def cosine_similarity(question_embeddings, embeddings):
    dot_product = np.dot(question_embeddings, embeddings)
    norm1 = np.linalg.norm(question_embeddings)
    norm2 = np.linalg.norm(embeddings)
    return dot_product/(norm1*norm2)

def find_top_chunks(question_embeddings, embeddings, chunks, top_n = 3):
    scores = []
    for i, chunk_emb in enumerate(embeddings):
        similarity = cosine_similarity(question_embeddings, chunk_emb)
        scores.append((similarity, chunks[i]))
    
    scores.sort(key=lambda x:x[0], reverse =True)
    return [chunk for score, chunk in scores[:top_n]]

st.title("PDF Assistant")
uploaded_file = st.file_uploader("Upload you PDF file here", type = "pdf")
if uploaded_file is not None:
    with st.spinner("Processing PDF....."):
        chunks, embeddings = process_pdf(uploaded_file)
    st.success(f"Ready! {len(chunks)} chunks processed")

    with st.form("question_form"):
        question = st.text_input("Ask a question about the document")
        submitted = st.form_submit_button("Ask")

    if submitted:
        if question:
            model = load_embedding_model()
            question_embeddings = model.encode(question)
            top_chunks = find_top_chunks(question_embeddings,embeddings,chunks, 3 )
            context = "\n\n".join(top_chunks)
            prompt = f"""You are a helpful assistant. Answer the question below using only the context given below.
            If you cant find the answer using only the given context say "I dont know" or "I don't know about it based on the given document. 
            Do not use any outside knowledge.
            Context:{context}
            Question:{question}
            """

            response = client.messages.create(
                model = "claude-sonnet-4-6",
                max_tokens = 1024,
                messages = [
                    {"role":"user", "content": prompt}
                ]
            )
            st.markdown(response.content[0].text)



        


