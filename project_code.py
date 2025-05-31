import streamlit as st
import pandas as pd
import os
import sqlite3
from datetime import datetime
import logging
import re

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Set it in your .env file.")
    st.stop()

# Database
DB_PATH = "chat_history.db"
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                patient_name TEXT,
                question TEXT,
                answer TEXT
            )
        """)
init_db()

# Load dataset
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if 'age_flag' in df.columns:
        df = df[df['age_flag'] == 'Valid']
    return df[["disease", "age", "gender", "symptoms", "medical_history", "test_results",
               "differential_diagnoses", "medications", "treatment_plan", "follow_ups"]]

df = load_data("synthetic_medical_data_genai_100k.csv")

# Format records
def format_row(row):
    base = (f"Disease: {row['disease']}\nAge: {row['age']}\nGender: {row['gender']}\n"
            f"Symptoms: {row['symptoms']}\nMedical History: {row['medical_history']}\n"
            f"Test Results: {row['test_results']}\nDifferential Diagnoses: {row['differential_diagnoses']}\n"
            f"Medications: {row['medications']}\nTreatment Plan: {row['treatment_plan']}\n"
            f"Follow-ups: {row['follow_ups']}")
    if row['age'] < 18:
        base += "\nPediatric Note: Adjust doses based on weight. Watch for Kawasaki, measles, etc."
    return base

# Build document chunks
@st.cache_resource
def build_vectorstore(df):
    docs = [Document(page_content=format_row(row),
                     metadata={"disease": row["disease"], "age": row["age"], "gender": row["gender"]})
            for _, row in df.iterrows()]
    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

vectorstore = build_vectorstore(df)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Prompt template
template = """
You are MediMind, an AI clinical assistant...

Context:
{context}

Question:
{question}

Helpful Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Model
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.0, max_tokens=1000)
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": lambda x: x,
    }
    | prompt
    | llm
)

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

def append_to_db(name, question, answer):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO chat_history (timestamp, patient_name, question, answer) VALUES (?, ?, ?, ?)",
                     (ts, name, question, answer))

# Streamlit UI
st.title("🧠 MediMind - Medical AI Assistant")

with st.form("query_form"):
    patient_name = st.text_input("Patient Name", "John Doe")
    query = st.text_area("Enter your medical query", "Patient Name: John Doe, 5-year-old male, weight 18 kg, symptoms: fever, rash, headache.")
    submitted = st.form_submit_button("Diagnose")

if submitted:
    st.markdown("## 🔍 MediMind's Diagnosis")
    age_match = re.search(r'(\d+)-year-old', query.lower())
    age = int(age_match.group(1)) if age_match else None
    gender_match = re.search(r'\b(male|female|other)\b', query.lower())
    gender = gender_match.group(1) if gender_match else None
    weight_match = re.search(r'weight (\d+\.?\d*) kg', query.lower())
    weight = float(weight_match.group(1)) if weight_match else None

    try:
        answer = rag_chain.invoke(query).content
    except Exception as e:
        st.error(f"RAG error: {e}")
        answer = "Sorry, something went wrong with the LLM."

    if "Sorry" in answer:
        # TF-IDF Fallback
        st.info("Using fallback TF-IDF method...")
        vectorizer = TfidfVectorizer()
        all_symptoms = df['symptoms'].str.lower().fillna('').tolist()
        query_vec = vectorizer.fit_transform([query.lower()] + all_symptoms)
        sims = cosine_similarity(query_vec[0:1], query_vec[1:])[0]
        top_ids = np.argsort(sims)[-3:]
        fallback_df = df.iloc[top_ids]

        # Filter by age/gender
        if age:
            fallback_df = fallback_df[fallback_df['age'].between(age-5, age+5)]
        if gender:
            fallback_df = fallback_df[fallback_df['gender'].str.lower() == gender]

        if fallback_df.empty:
            answer = "No similar patient records found."
        else:
            responses = []
            for _, row in fallback_df.iterrows():
                meds = row['medications']
                if row['age'] < 18 and weight:
                    if 'paracetamol' in meds.lower():
                        meds = meds.replace('paracetamol 500 mg', f'paracetamol {15 * weight:.0f} mg')
                    if 'ibuprofen' in meds.lower():
                        meds = meds.replace('ibuprofen 400 mg', f'ibuprofen {10 * weight:.0f} mg')

                responses.append(f"""
**Disease**: {row['disease']}
**Symptoms**: {row['symptoms']}
**Medications**: {meds}
**Treatment Plan**: {row['treatment_plan']}
**Follow-ups**: {row['follow_ups']}
""")
            answer = "\n\n".join(responses)

    st.markdown(answer)
    append_to_db(patient_name, query, answer)

# Show past chat history
with st.expander("📜 View Chat History"):
    with sqlite3.connect(DB_PATH) as conn:
        hist_df = pd.read_sql_query("SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT 10", conn)
    st.dataframe(hist_df)
