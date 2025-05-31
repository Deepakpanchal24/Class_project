import pandas as pd
import sqlite3
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from datetime import datetime

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("Error: GROQ_API_KEY not found in .env file. Please set it.")
    exit(1)

# Initialize SQLite database for chat history
CHAT_HISTORY_DB = "chat_history.db"

def init_db():
    try:
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
    except Exception as e:
        print(f"Error initializing SQLite database: {str(e)}")
        exit(1)
    finally:
        conn.close()

init_db()

# Load embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading embedding model: {str(e)}")
    exit(1)

# Load CSV data and create FAISS index
def load_data(csv_path):
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file {csv_path} not found in {os.getcwd()}.")
        df = pd.read_csv(csv_path)
        required_columns = ['disease', 'symptoms', 'treatment_plan']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        texts = [f"Disease: {row['disease']}\nSymptoms: {row['symptoms']}\nTreatment: {row['treatment_plan']}"
                 for _, row in df.iterrows()]
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, texts, df
    except Exception as e:
        print(f"Error loading CSV or creating FAISS index: {str(e)}")
        exit(1)

csv_path = "MediMind_cleaned_utf8.csv"
try:
    faiss_index, texts, csv_data = load_data(csv_path)
except Exception as e:
    print(f"Failed to load data: {str(e)}")
    exit(1)

# Initialize Groq model
try:
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.0, max_tokens=1000)
except Exception as e:
    print(f"Error initializing Groq model: {str(e)}. Verify GROQ_API_KEY.")
    exit(1)

# Define prompt template
template = """
You are MediMind, an AI-driven clinical decision support assistant for physicians. Your role is to analyze patient-specific data from emergency, primary care, and pediatric settings to provide both Symptom Analysis and Clinical Decision Support.

You will receive structured clinical input including patient age, gender, symptoms, medical history, test results, and physician queries from CSV and PDF data sources.

[OBJECTIVES]:
- Perform a concise [Symptom Analysis]:
  - List up to 4 [differential diagnoses] with estimated [probabilities].
  - Use evidence from symptoms, history, and labs/imaging to support each diagnosis.
  - Highlight any [urgent alerts] (e.g., STEMI, Kawasaki disease, sepsis).

- Provide a detailed [Clinical Decision Support] plan:
  - Recommend [immediate actions]: medications (with dosages), labs, referrals, or procedures.
  - Propose a structured [treatment plan] (medical + lifestyle/supportive care).
  - Suggest [follow-up actions]: timeline for reassessment, repeat tests, or specialist input.
  - End with a [concise documentation summary] for clinical notes or EHR.

[PEDIATRIC CARE]:
- For children under 18, adjust drug dosages (e.g., ibuprofen 10 mg/kg) based on weight if available.
- Consider pediatric-specific diseases and guidance.

[RULES]:
- Use only the provided patient context and real cases from CSV and PDF sources.
- Prioritize cases that match age, gender, symptoms, and lab findings.
- Do not generate hypothetical diseases or treatments not supported by source data.
- If no matching records are found: reply “Sorry, no patient records found matching your criteria in the dataset. Consider general medical evaluation.”

[FORMAT YOUR RESPONSE EXACTLY AS]:
## [Symptom Analysis]
- [Differential Diagnoses]:
  - [Diagnosis 1]: [probability]% – [supporting reasoning]
  - [Diagnosis 2]: [probability]% – [supporting reasoning]
  - [Diagnosis 3]: [probability]% – [supporting reasoning]
  - [Diagnosis 4]: [probability]% – [supporting reasoning]

- [Alert]:
  - [Red-flag condition and immediate concern]

## [Clinical Decision Support]
- [Immediate Actions]:
  - [List meds (with dosage), procedures, urgent labs, referrals]

- [Treatment Plan]:
  - [Step-by-step treatment with lifestyle/supportive components]

- [Follow-Ups]:
  - [Tests, reassessments, and timeline]

- [Documentation]:
  - [Bullet-form summary for EHR or physician notes]

Context:
{context}

Question:
{question}

Helpful Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Function to search similar cases
def search_similar_cases(query):
    try:
        if not query or query.strip().lower() in ["question", ""]:
            raise ValueError("Query is empty or invalid (e.g., 'question'). Please provide a meaningful query.")
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, k=3)
        if not indices[0].size:
            raise ValueError("No matching cases found in the dataset.")
        return [texts[i] for i in indices[0]]
    except Exception as e:
        print(f"Error in similarity search: {str(e)}")
        return []

# Function to save chat history
def save_chat_history(patient_name, query, answer):
    try:
        conn = sqlite3.connect(CHAT_HISTORY_DB)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO chat_history (timestamp, patient_name, question, answer) VALUES (?, ?, ?, ?)",
            (timestamp, patient_name, query, answer)
        )
        conn.commit()
    except Exception as e:
        print(f"Error saving to SQLite database: {str(e)}")
    finally:
        conn.close()

# Function to view chat history
def view_chat_history():
    try:
        conn = sqlite3.connect(CHAT_HISTORY_DB)
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
        print(f"Error reading chat history: {str(e)}")
        return f"Error reading chat history: {str(e)}."
    finally:
        conn.close()

# Function to clear input fields
def clear_inputs():
    return "", "", "", "Male", "", "", "", "", ""

# Main diagnosis function
def diagnose_patient(patient_name, age, weight, gender, symptoms, medical_history, test_results, physician_query):
    if not patient_name.strip():
        return "Please provide a valid patient name."
    if not physician_query.strip():
        return "Please provide a valid physician query."
    try:
        # Validate age
        try:
            age = int(age.strip()) if age.strip() else -1
            if age < 0 or age > 120:
                return "Please enter a valid age between 0 and 120."
        except ValueError:
            return "Please enter a valid integer for age."

        # Validate weight
        try:
            weight = float(weight.strip()) if weight.strip() else -1
            if weight <= 0 or weight > 500:
                return "Please enter a valid weight between 0 and 500 kg."
        except ValueError:
            return "Please enter a valid number for weight."

        # Validate gender
        if gender not in ["Male", "Female", "Other"]:
            return "Please select a valid gender from the dropdown."

        # Combine all inputs for search
        symptoms = symptoms.strip() if symptoms else "None"
        medical_history = medical_history.strip() if medical_history else "None"
        test_results = test_results.strip() if test_results else "None"
        full_query = (
            f"Patient: {patient_name}\n"
            f"Age: {age} years\n"
            f"Weight: {weight} kg\n"
            f"Gender: {gender}\n"
            f"Symptoms: {symptoms}\n"
            f"Medical History: {medical_history}\n"
            f"Test Results: {test_results}\n"
            f"Physician Query: {physician_query}"
        )
        context = "\n\n".join(search_similar_cases(full_query))
        if not context:
            return "Sorry, no patient records found matching your criteria in the dataset. Consider general medical evaluation."
        response = llm.invoke(prompt.format(context=context, question=full_query)).content
        save_chat_history(patient_name, full_query, response)
        return response
    except Exception as e:
        print(f"Error in diagnose_patient: {str(e)}")
        return f"Error processing query: {str(e)}. Please check inputs and try again."

# Gradio interface
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
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear")
            output = gr.Textbox(label="MediMind Diagnostic Response", lines=15)
            submit_btn.click(
                fn=diagnose_patient,
                inputs=[patient_name, age, weight, gender, symptoms, medical_history, test_results, physician_query],
                outputs=output
            )
            clear_btn.click(
                fn=clear_inputs,
                inputs=[],
                outputs=[patient_name, age, weight, gender, symptoms, medical_history, test_results, physician_query, output]
            )
        with gr.TabItem("Chat History"):
            history_output = gr.Textbox(label="Chat History", lines=10, interactive=False)
            refresh_btn = gr.Button("Refresh History")
            refresh_btn.click(
                fn=view_chat_history,
                inputs=[],
                outputs=history_output
            )

if __name__ == "__main__":
    try:
        print("Launching Gradio on port 7861...")
        iface.launch(server_port=7861, server_name="0.0.0.0")
        print("Successfully launched on port 7861. Access at http://127.0.0.1:7861")
    except OSError as e:
        print(f"Failed to launch on port 7861: {str(e)}")
        print("Run 'netstat -aon | findstr :7861' to check port usage.")
        print("Use 'taskkill /PID <PID> /F' to free the port.")
        exit(1)