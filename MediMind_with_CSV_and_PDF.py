import pandas as pd
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import os
import re
import logging
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
import PyPDF2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in .env file. Please set it in D:\\test5 Sqlite\\.env.")
    exit(1)

# SQLite database setup
CHAT_HISTORY_DB = "chat_history.db"

def init_db():
    """Initialize SQLite database and create chat_history table if it doesn't exist."""
    try:
        with sqlite3.connect(CHAT_HISTORY_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL
                )
            """)
            conn.commit()
            logger.info(f"Initialized SQLite database: {CHAT_HISTORY_DB}")
    except Exception as e:
        logger.error(f"Error initializing SQLite database: {str(e)}")
        exit(1)

# Call init_db at startup
init_db()

# Function to extract text from PDF and parse into a DataFrame
def parse_pdf_to_dataframe(pdf_file):
    """Extract text from PDF and convert to a DataFrame with the same columns as CSV."""
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            logger.info(f"Extracted text from {pdf_file}.")
    except Exception as e:
        logger.error(f"Error reading PDF file {pdf_file}: {str(e)}")
        return pd.DataFrame()

    # Define expected columns
    columns = ["disease", "age", "gender", "symptoms", "medical_history", "test_results",
               "differential_diagnoses", "medications", "treatment_plan", "follow_ups"]
    records = []

    # Split text into potential patient records (assuming each record is separated by double newlines)
    patient_blocks = text.split("\n\n")
    for block in patient_blocks:
        record = {col: "None" for col in columns}  # Default values
        try:
            # Use regex to extract fields based on keywords
            for field in columns:
                pattern = rf"{field.replace('_', ' ').title()}: (.*?)(?:\n|$)"
                match = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    if field == "age":
                        try:
                            value = int(value) if value.isdigit() else "None"
                        except:
                            value = "None"
                    record[field] = value
            records.append(record)
        except Exception as e:
            logger.warning(f"Error parsing patient block: {str(e)}")
            continue

    if not records:
        logger.warning(f"No valid patient records extracted from {pdf_file}.")
        return pd.DataFrame()

    df = pd.DataFrame(records, columns=columns)
    logger.info(f"Parsed {len(df)} patient records from {pdf_file}.")
    return df

# Load and combine CSV and PDF datasets
csv_file = "MediMind_cleaned_utf8.csv"
pdf_file = "Diagnostic.pdf"
try:
    # Load CSV
    df_csv = pd.read_csv(csv_file, encoding='utf-8')
    logger.info(f"Successfully loaded {csv_file} with {len(df_csv)} rows.")
except Exception as e:
    logger.error(f"Error loading {csv_file}: {str(e)}. Ensure file exists in D:\\test5 Sqlite.")
    exit(1)

# Load and parse PDF
df_pdf = parse_pdf_to_dataframe(pdf_file)

# Combine datasets
if not df_pdf.empty:
    # Ensure PDF DataFrame has the same columns as CSV
    required_columns = ["disease", "age", "gender", "symptoms", "medical_history", "test_results",
                        "differential_diagnoses", "medications", "treatment_plan", "follow_ups"]
    df_pdf = df_pdf.reindex(columns=required_columns, fill_value="None")
    data = pd.concat([df_csv, df_pdf], ignore_index=True)
else:
    logger.warning("No data extracted from PDF. Using only CSV data.")
    data = df_csv

# Filter out rows with invalid ages (if age_flag exists in CSV)
if 'age_flag' in data.columns:
    data = data[data['age_flag'] == 'Valid'][["disease", "age", "gender", "symptoms", "medical_history", "test_results",
                                             "differential_diagnoses", "medications", "treatment_plan", "follow_ups"]].copy()
else:
    data = data[["disease", "age", "gender", "symptoms", "medical_history", "test_results",
                 "differential_diagnoses", "medications", "treatment_plan", "follow_ups"]].copy()
logger.info(f"Combined dataset has {len(data)} rows after filtering.")

# Function to format patient records into a string with pediatric note
def format_row(row):
    base_format = (f"Disease: {row['disease']}\n"
                   f"Age: {row['age']}\n"
                   f"Gender: {row['gender']}\n"
                   f"Symptoms: {row['symptoms']}\n"
                   f"Medical History: {row['medical_history']}\n"
                   f"Test Results: {row['test_results']}\n"
                   f"Differential Diagnoses: {row['differential_diagnoses']}\n"
                   f"Medications: {row['medications']}\n"
                   f"Treatment Plan: {row['treatment_plan']}\n"
                   f"Follow-ups: {row['follow_ups']}")
    try:
        age = float(row['age']) if isinstance(row['age'], (int, float, str)) and str(row['age']).replace('.', '', 1).isdigit() else None
        if age is not None and age < 18:
            base_format += f"\nPediatric Note: Adjust dosages for age/weight (e.g., paracetamol 15 mg/kg); consider pediatric-specific conditions like Kawasaki disease or measles."
    except Exception as e:
        logger.warning(f"Error processing age for pediatric note: {str(e)}")
    return base_format

# Create documents for each patient record with metadata
docs = [Document(page_content=format_row(row),
                metadata={"disease": row["disease"], "age": row["age"], "gender": row["gender"]})
        for _, row in data.iterrows()]
logger.info(f"Created {len(docs)} documents from combined dataset.")

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
chunks = splitter.split_documents(docs)
logger.info(f"Split into {len(chunks)} chunks.")

# Initialize embeddings and vectorstore
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("faiss_index_medical")
    logger.info("Vectorstore created and saved as faiss_index_medical.")
except Exception as e:
    logger.error(f"Error creating vectorstore: {str(e)}")
    exit(1)

# Load the LLM model (Groq)
try:
    from langchain_groq import ChatGroq
    ChatGroq.model_rebuild()  # Rebuild the model to resolve BaseCache dependency
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=groq_api_key,
        temperature=0.0,
        max_tokens=1000
    )
    logger.info("Groq model initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Groq model: {str(e)}. Verify GROQ_API_KEY is valid.")
    exit(1)

# Define the prompt template with pediatric guidance
template = """
You are MediMind, an AI-driven clinical decision support assistant specializing in Symptom Analysis and Differential Diagnosis across emergency, primary care, and pediatric settings.

You will receive patient-specific context including disease, age, gender, symptoms, medical history, and test results. Your job is to help physicians analyze this information and provide a focused diagnostic and treatment plan.

🧠 TASK:
- Identify up to 4 **differential diagnoses** with associated **probabilities** based on context.
- Clearly **highlight urgent alerts** or red-flag features (e.g., STEMI, meningitis, Kawasaki disease).
- Recommend **immediate clinical actions** including labs, medications (with dosage), or referrals.
- Present a **structured treatment plan** and supportive care instructions.
- Suggest **follow-up actions**, repeat tests, and timeline for reassessment.
- Add a **concise documentation summary** for inclusion in the patient’s record.

🧒 PEDIATRIC NOTE:
- For patients <18 years, adjust medication dosages (e.g., ibuprofen 10 mg/kg, paracetamol 15 mg/kg) based on weight if provided, and include pediatric-specific considerations (e.g., Kawasaki, roseola, measles).

🧪 DIAGNOSTIC SUPPORT RULES:
- Use only the provided patient context and historical cases.
- Prioritize age/gender match when filtering records.
- If no records match, say: "Sorry, no patient records found matching your criteria in the dataset. Consider general medical evaluation."
- NEVER invent diseases or treatments not found in the source data.

📝 FORMAT YOUR RESPONSE AS:
**Differential Diagnoses**:
[List with probabilities]

**Alerts**:
[Urgent red flags or conditions needing immediate action]

**Immediate Actions**:
[Procedures, medications with dosage, labs, or referrals]

**Treatment Plan**:
[Medical + supportive plan including lifestyle or specialist care]

**Follow-Ups**:
[Timeline and actions for reassessment]

**Documentation**:
[Bullet summary for EHR or clinical notes]

Context:
{context}

Question:
{question}

Helpful Answer:
"""


# Initialize prompt
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Initialize the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Set up the RAG chain
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

# Initialize conversation memory
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Function to append to chat history SQLite
def append_to_chat_history_sqlite(patient_name, question, answer):
    """Append chat history to SQLite database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with sqlite3.connect(CHAT_HISTORY_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (timestamp, patient_name, question, answer)
                VALUES (?, ?, ?, ?)
            """, (timestamp, patient_name, question.replace('\n', ' ').replace('\r', ' '),
                  answer.replace('\n', ' ').replace('\r', ' ')))
            conn.commit()
            logger.info(f"Appended chat history for {patient_name} to SQLite database.")
    except Exception as e:
        logger.error(f"Error writing to SQLite database: {str(e)}")

# Medical diagnosis tool
@tool
def medical_diagnosis_tool(query: str) -> str:
    """Search patient records to provide differential diagnoses, treatments, and follow-ups based on physician query."""
    global data, memory

    # Extract age, gender, weight, and patient name from query for filtering
    age_match = re.search(r'(\d+)-year-old', query.lower())
    gender_match = re.search(r'\b(male|female|other)\b', query.lower())
    weight_match = re.search(r'weight (\d+\.?\d*) kg', query.lower())
    name_match = re.search(r'patient name: ([a-zA-Z\s]+)', query.lower())
    age_filter = int(age_match.group(1)) if age_match else None
    gender_filter = gender_match.group(1) if gender_match else None
    weight = float(weight_match.group(1)) if weight_match else None
    patient_name = name_match.group(1).strip() if name_match else "Unknown"

    # Incorporate chat history from memory
    history = memory.load_memory_variables({})["chat_history"]
    history_text = "\n".join([f"Q: {msg.content}" for msg in history[::2]] +
                            [f"A: {msg.content}" for msg in history[1::2]])
    refined_query = query + "\nPrevious chats:\n" + history_text

    # Run the RAG chain
    try:
        result = rag_chain.invoke(refined_query)
        if "Sorry" in result.content:
            logger.warning("RAG chain returned no matches, falling back to dataset filtering.")
    except Exception as e:
        logger.error(f"Error in RAG chain: {str(e)}")
        result = "Error processing query. Please try again."

    # Fallback: Filter dataset with TF-IDF similarity
    if isinstance(result, str) or "Sorry" in result.content:
        try:
            if data.empty:
                logger.warning("Dataset is empty, cannot perform TF-IDF fallback.")
                return "No patient records available in the dataset. Please ensure data is loaded correctly."
            vectorizer = TfidfVectorizer()
            all_symptoms = data['symptoms'].str.lower().fillna('').tolist()
            query_vec = vectorizer.fit_transform([query.lower()] + all_symptoms)
            similarities = cosine_similarity(query_vec[0:1], query_vec[1:])[0]
            top_indices = np.argsort(similarities)[-3:]  # Top 3 matches
            filtered_records = data.iloc[top_indices]

            if age_filter:
                filtered_records = filtered_records[filtered_records['age'].apply(lambda x: isinstance(x, (int, float)) and (age_filter - 5 <= x <= age_filter + 5))]
            if gender_filter:
                filtered_records = filtered_records[filtered_records['gender'].str.lower() == gender_filter]

            if not filtered_records.empty:
                result = ""
                urgent_conditions = {
                    'stroke': 'Immediate specialist consult recommended',
                    'myocardial_infarction': 'Immediate specialist consult recommended',
                    'meningitis': 'Urgent neurology consult for headache, fever, rash',
                    'anaphylaxis': 'Urgent ER visit for rash, dizziness, nausea'
                }

                for _, row in filtered_records.iterrows():
                    try:
                        diagnoses = json.loads(row['differential_diagnoses']) if row['differential_diagnoses'] and isinstance(row['differential_diagnoses'], str) else []
                        if not isinstance(diagnoses, list):
                            diagnoses = []
                    except Exception as e:
                        logger.warning(f"Invalid diagnoses format for record: {row['disease']}, error: {str(e)}")
                        diagnoses = []

                    # Adjust probabilities based on symptom overlap
                    query_symptoms = set(query.lower().split())
                    record_symptoms = set(row['symptoms'].lower().split('; ')) if row['symptoms'] and isinstance(row['symptoms'], str) else set()
                    overlap = len(query_symptoms.intersection(record_symptoms)) / len(query_symptoms) if query_symptoms else 1
                    formatted_diagnoses = '; '.join([f"{d.get('name', 'Unknown')}: {d.get('probability', 0)*100*overlap:.0f}%" for d in diagnoses]) if diagnoses else "No diagnoses available"

                    # Check for urgent conditions
                    alert = "Monitor for complications"
                    for condition, message in urgent_conditions.items():
                        if row['disease'] and condition in row['disease'].lower() or (
                            condition == 'meningitis' and 'headache' in query.lower() and 'fever' in query.lower() and 'rash' in query.lower()) or (
                            condition == 'anaphylaxis' and 'rash' in query.lower() and 'dizziness' in query.lower()):
                            alert = message
                            break

                    # Adjust medications for pediatric patients
                    medications = row['medications'] if row['medications'] and isinstance(row['medications'], str) else "None"
                    try:
                        age = float(row['age']) if isinstance(row['age'], (int, float, str)) and str(row['age']).replace('.', '', 1).isdigit() else None
                        if age is not None and age < 18 and weight:
                            if 'paracetamol' in medications.lower():
                                medications = medications.replace('paracetamol 500 mg', f'paracetamol {15 * weight:.0f} mg')
                            if 'ibuprofen' in medications.lower():
                                medications = medications.replace('ibuprofen 400 mg', f'ibuprofen {10 * weight:.0f} mg')
                    except:
                        pass

                    result += (
                        f"**Patient**: {patient_name}\n"
                        f"**Differential Diagnoses**:\n"
                        f"{formatted_diagnoses}\n"
                        f"**Alerts**: {alert}\n"
                        f"**Immediate Actions**:\n{medications}\n"
                        f"**Treatment Plan**:\n{row['treatment_plan'] if row['treatment_plan'] and isinstance(row['treatment_plan'], str) else 'None'}\n"
                        f"**Follow-Ups**:\n{row['follow_ups'] if row['follow_ups'] and isinstance(row['follow_ups'], str) else 'None'}\n"
                        f"**Documentation**:\nDiagnosed {row['disease'] if row['disease'] else 'unknown'}; initiated {medications.split(';')[0] if medications else 'no medications'}; planned {row['treatment_plan'].split(';')[0] if row['treatment_plan'] and isinstance(row['treatment_plan'], str) else 'no plan'}.\n\n"
                    )
            else:
                result = f"Sorry, no patient records found for {patient_name} matching your criteria in the dataset. Consider general medical evaluation."
        except Exception as e:
            logger.error(f"Error in fallback mechanism: {str(e)}")
            result = "Error processing query in fallback. Please try again."

    # Save to conversation memory and SQLite
    response_content = result.content if hasattr(result, 'content') else str(result)
    memory.save_context({"input": query}, {"output": response_content})
    append_to_chat_history_sqlite(patient_name, query, response_content)
    return response_content

# Initialize agent with tools
json_responder = initialize_agent(
    tools=[medical_diagnosis_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory
)

# Gradio Interface with Tabs
def diagnose_patient(patient_name, age, weight, gender, symptoms, medical_history, test_results, physician_query):
    """Process patient diagnosis query with input validation."""
    # Input validation
    if not patient_name.strip():
        logger.warning("Invalid patient name provided.")
        return "Please enter a valid patient name."
    
    try:
        age = int(age.strip()) if age.strip() else -1
        if age < 0 or age > 120:
            logger.warning(f"Invalid age provided: {age}")
            return "Please enter a valid age between 0 and 120 years."
    except ValueError:
        logger.warning(f"Non-integer age provided: {age}")
        return "Please enter a valid integer for age."
    
    try:
        weight = float(weight.strip()) if weight.strip() else -1
        if weight < 0 or weight > 300:
            logger.warning(f"Invalid weight provided: {weight}")
            return "Please enter a valid weight between 0 and 300 kg."
    except ValueError:
        logger.warning(f"Non-numeric weight provided: {weight}")
        return "Please enter a valid number for weight."
    
    if not gender or gender not in ["Male", "Female", "Other"]:
        logger.warning(f"Invalid gender provided: {gender}")
        return "Please select a valid gender (Male, Female, or Other)."
    
    if not physician_query.strip():
        logger.warning("No physician query provided.")
        return "Please enter a physician query."
    
    # Truncate optional fields to prevent excessive length
    symptoms = (symptoms[:500] if symptoms else "None").strip()
    medical_history = (medical_history[:500] if medical_history else "None").strip()
    test_results = (test_results[:500] if test_results else "None").strip()
    patient_name = patient_name[:100].strip()

    try:
        # Construct the query string from inputs
        query = (f"Patient name: {patient_name}; {age}-year-old {gender} patient, weight {weight} kg, "
                 f"with symptoms: {symptoms}; medical history: {medical_history}; test results: {test_results}; "
                 f"physician query: {physician_query}")
        response = medical_diagnosis_tool.invoke({"query": query})
        response_content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Processed query for {patient_name}: {query[:50]}... Response: {response_content[:50]}...")
        return response_content
    except Exception as e:
        logger.error(f"Error in diagnose_patient for {patient_name}: {str(e)}")
        return f"Error processing query for {patient_name}: {str(e)}. Please check logs and try again."

def view_chat_history():
    """Retrieve chat history from SQLite database."""
    try:
        with sqlite3.connect(CHAT_HISTORY_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, patient_name, question, answer FROM chat_history ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            formatted_history = ""
            for index, row in enumerate(rows):
                formatted_history += (
                    f"**Timestamp**: {row[0]}\n"
                    f"**Patient**: {row[1]}\n"
                    f"**Question {index + 1}**: {row[2]}\n"
                    f"**Answer**: {row[3]}\n\n"
                )
            return formatted_history if rows else "No chat history available."
    except Exception as e:
        logger.error(f"Error reading from SQLite database: {str(e)}")
        return f"Error reading chat history: {str(e)}. Please check logs."

# Gradio Interface
with gr.Blocks() as iface:
    gr.Markdown("# MediMind Diagnostic Assistant")
    
    with gr.Tabs():
        with gr.TabItem("Diagnosis"):
            gr.Markdown("Enter patient details and query below:")
            patient_name = gr.Textbox(label="Patient Name", placeholder="E.g., John Doe")
            age = gr.Textbox(label="Age (years)", placeholder="E.g., 52")
            weight = gr.Textbox(label="Weight (kg)", placeholder="E.g., 70")
            gender = gr.Dropdown(label="Gender", choices=["Male", "Female", "Other"], value="Male")
            symptoms = gr.Textbox(label="Symptoms", placeholder="E.g., chest pain, shortness of breath")
            medical_history = gr.Textbox(label="Medical History", placeholder="E.g., hypertension, diabetes")
            test_results = gr.Textbox(label="Test Results", placeholder="E.g., ECG normal, elevated troponin")
            physician_query = gr.Textbox(label="Physician Query", placeholder="E.g., What's the likely diagnosis?")
            submit_btn = gr.Button("Submit")
            output = gr.Textbox(label="MediMind Diagnostic Response")
            submit_btn.click(
                fn=diagnose_patient,
                inputs=[patient_name, age, weight, gender, symptoms, medical_history, test_results, physician_query],
                outputs=output
            )
        
        with gr.TabItem("Chat History"):
            gr.Markdown("View the conversation history below:")
            history_output = gr.Textbox(label="Chat History", lines=10, interactive=False)
            refresh_btn = gr.Button("Refresh History")
            refresh_btn.click(
                fn=view_chat_history,
                inputs=[],
                outputs=history_output
            )

# Run the app
if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    try:
        iface.launch(server_port=7860)
    except Exception as e:
        logger.error(f"Error launching Gradio: {str(e)}. Try a different port (e.g., 7861).")
        exit(1)