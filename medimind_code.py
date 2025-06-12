import pandas as pd
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import os
import sqlite3
import fitz  # PyMuPDF
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("Error: GROQ_API_KEY not found in .env file.")
    exit(1)

# SQLite database setup
CHAT_HISTORY_DB = "chat_history.db"

def init_db():
    """Set up SQLite database for chat history."""
    conn = sqlite3.connect(CHAT_HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            patient_name TEXT,
            question TEXT,
            answer TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("SQLite database initialized.")

init_db()

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Step 1: Load CSV data
def load_csv(csv_path):
    """Load CSV and format rows into text."""
    df = pd.read_csv(csv_path)
    texts = []
    for _, row in df.iterrows():
        text = (f"Disease: {row['disease']}\n"
                f"Age: {row['age']}\n"
                f"Gender: {row['gender']}\n"
                f"Symptoms: {row['symptoms']}\n"
                f"Medical History: {row['medical_history']}\n"
                f"Test Results: {row['test_results']}\n"
                f"Differential Diagnoses: {row['differential_diagnoses']}\n"
                f"Medications: {row['medications']}\n"
                f"Treatment Plan: {row['treatment_plan']}\n"
                f"Follow-ups: {row['follow_ups']}")
        texts.append(text)
    print(f"Loaded {len(df)} rows from CSV.")
    return texts, df

# Step 2: Load PDF data
def load_pdf(pdf_path):
    """Load text from PDF pages."""
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc if page.get_text().strip()]
    print(f"Loaded {len(texts)} pages from PDF.")
    return texts

# Step 3: Create documents and vectorstore
def create_vectorstore(csv_texts, pdf_texts, csv_data):
    """Create FAISS vectorstore from CSV and PDF texts."""
    # Create documents for CSV
    csv_docs = [
        Document(
            page_content=text,
            metadata={"source": "csv", "disease": row["disease"], "age": row["age"], "gender": row["gender"]}
        ) for text, (_, row) in zip(csv_texts, csv_data.iterrows())
    ]
    # Create documents for PDF
    pdf_docs = [Document(page_content=text, metadata={"source": "pdf"}) for text in pdf_texts]
    all_docs = csv_docs + pdf_docs
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    
    # Create and save vectorstore
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("faiss_index_medical")
    print(f"Created vectorstore with {len(chunks)} chunks.")
    return vectorstore

# Load data
csv_path = "MediMind_cleaned_utf8.csv"  # Replace with your CSV path
pdf_path = "Diagnostic.pdf"  # Replace with your PDF path

csv_texts, csv_data = load_csv(csv_path)
pdf_texts = load_pdf(pdf_path)
vectorstore = create_vectorstore(csv_texts, pdf_texts, csv_data)

# Initialize Grok model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0.0,
    max_tokens=1000
)
print("Grok model initialized.")

# Define prompt template
prompt_template = """
You are MediMind, a medical assistant for symptom analysis and diagnosis.
Based on the patient data and context, provide:
- Up to 4 differential diagnoses with probabilities.
- Urgent alerts for critical conditions.
- Immediate actions (medications, labs, referrals).
- Treatment plan and follow-ups.
- Documentation summary for patient records.

For patients under 18, adjust medication dosages (e.g., paracetamol 15 mg/kg).

Context:
{context}

Question:
{question}

Answer:
**Differential Diagnoses**:
[List with probabilities]

**Alerts**:
[Urgent conditions]

**Immediate Actions**:
[Medications, labs, referrals]

**Treatment Plan**:
[Medical and supportive care]

**Follow-Ups**:
[Timeline and reassessment]

**Documentation**:
[Summary for records]
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# Set up RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    }
    | prompt
    | llm
)

# Save chat history to SQLite
def save_chat_history(patient_name, question, answer):
    """Save question and answer to SQLite database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(CHAT_HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (timestamp, patient_name, question, answer) VALUES (?, ?, ?, ?)",
        (timestamp, patient_name, question, answer)
    )
    conn.commit()
    conn.close()
    print(f"Saved chat history for {patient_name}.")

# Diagnosis function
def diagnose_patient(patient_name, age, weight, gender, symptoms, medical_history, test_results, query):
    """Generate diagnosis based on patient input."""
    if not patient_name or not query:
        return "Please provide patient name and query."
    
    try:
        age = int(age)
        weight = float(weight)
        if age < 0 or weight < 0:
            return "Please provide valid age and weight."
    except ValueError:
        return "Please provide numeric age and weight."
    
    if gender not in ["Male", "Female", "Other"]:
        return "Please select a valid gender."
    
    input_query = (f"Patient name: {patient_name}; {age}-year-old {gender} patient, weight {weight} kg, "
                   f"symptoms: {symptoms or 'None'}; medical history: {medical_history or 'None'}; "
                   f"test results: {test_results or 'None'}; physician query: {query}")
    
    response = rag_chain.invoke(input_query)
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    save_chat_history(patient_name, input_query, response_text)
    return response_text

# View chat history
def view_chat_history():
    """Retrieve chat history from SQLite."""
    conn = sqlite3.connect(CHAT_HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, patient_name, question, answer FROM chat_history ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "No chat history available."
    
    history = ""
    for row in rows:
        history += (f"**Timestamp**: {row[0]}\n"
                    f"**Patient**: {row[1]}\n"
                    f"**Question**: {row[2]}\n"
                    f"**Answer**: {row[3]}\n\n")
    return history

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# MediMind Diagnostic Assistant")
    
    with gr.Tabs():
        with gr.TabItem("Diagnosis"):
            patient_name = gr.Textbox(label="Patient Name", placeholder="E.g., John Doe")
            age = gr.Textbox(label="Age (years)", placeholder="E.g., 52")
            weight = gr.Textbox(label="Weight (kg)", placeholder="E.g., 70")
            gender = gr.Dropdown(label="Gender", choices=["Male", "Female", "Other"], value="Male")
            symptoms = gr.Textbox(label="Symptoms", placeholder="E.g., chest pain")
            medical_history = gr.Textbox(label="Medical History", placeholder="E.g., hypertension")
            test_results = gr.Textbox(label="Test Results", placeholder="E.g., ECG normal")
            query = gr.Textbox(label="Physician Query", placeholder="E.g., What's the diagnosis?")
            submit_btn = gr.Button("Diagnose")
            output = gr.Textbox(label="Diagnosis", lines=10)
            submit_btn.click(
                fn=diagnose_patient,
                inputs=[patient_name, age, weight, gender, symptoms, medical_history, test_results, query],
                outputs=output
            )
        
        with gr.TabItem("Chat History"):
            history_output = gr.Textbox(label="History", lines=10, interactive=False)
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(fn=view_chat_history, inputs=[], outputs=history_output)

# Launch the app
if __name__ == "__main__":
    print("Starting Gradio app...")
    app.launch(server_port=7860)