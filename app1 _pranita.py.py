# # ------------------------------
# # 0. MUST be first Streamlit command
# # ------------------------------
# import streamlit as st
# st.set_page_config(page_title="🩺 MediMind Clinical Assistant")

# # ------------------------------
# # 1. Imports and environment
# # ------------------------------
# from dotenv import load_dotenv
# import os

# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI

# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     st.error("OPENAI_API_KEY not found. Please add it to your .env file.")
#     st.stop()
# os.environ["OPENAI_API_KEY"] = openai_api_key

# # ------------------------------
# # 2. Load FAISS vector store
# # ------------------------------
# @st.cache_resource
# def load_vectorstore():
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#     return FAISS.load_local("faiss_index_1", embeddings, allow_dangerous_deserialization=True)

# vectorstore = load_vectorstore()

# # ------------------------------
# # 3. Prompt template for RAG
# # ------------------------------
# prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a clinical decision support assistant. Use the retrieved context and outpatient guidelines (ADA, NICE, WHO) to provide a comprehensive diagnosis and treatment plan.

# Context:
# {context}

# Question:
# {question}

# Instructions:
# - ONLY recommend outpatient-safe diagnosis and treatment. If any red flags are detected, mention them clearly.
# - Follow evidence-based outpatient guidelines.
# - Fill in EVERY section below with complete, detailed information.
# - DO NOT skip any section.

# Format your response like this:

# **1. Symptom Analysis**
# - Differential Diagnoses (include % likelihood + 1-line reasoning per item)
# - Red Flags (list if any, else write "None")

# **2. Clinical Decision Support**
# - Immediate Outpatient Actions:
#     • Vitals to check
#     • Labs to order (e.g., HbA1c, CBC, TSH, etc.)
#     • Physical assessments
# - Medications:
#     • Name
#     • Dose
#     • Frequency
#     • Purpose
# - Lifestyle Modifications:
#     • Diet changes (e.g., low glycemic index, reduced carbs)
#     • Exercise (type, intensity, duration per week)
#     • Smoking cessation if applicable
# - Follow-Up Plan:
#     • When to repeat labs
#     • When to schedule next visit
#     • Any specialist referrals needed
# - Documentation Notes:
#     • Summary of findings
#     • Diagnosis and plan in 1-2 sentences

# Be concise but complete. Structure clearly. Do not leave any field blank.
# """
# )

# # ------------------------------
# # 4. Initialize LLM & QA Chain
# # ------------------------------
# llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
# clinical_qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectorstore.as_retriever(),
#     chain_type_kwargs={"prompt": prompt_template},
#     return_source_documents=False
# )

# # ------------------------------
# # 5. Streamlit UI
# # ------------------------------
# st.title("🩺 MediMind: Outpatient Clinical Assistant")

# query_text = st.text_area("Enter your clinical scenario:", height=250, value="""
# Context: Primary Care – Routine Visit for Fatigue
# Patient Demographics: 45-year-old female
# Symptoms: Fatigue for 3 months, frequent urination, increased thirst, 8-pound weight loss
# Medical History: Obesity (BMI 32), family history of Type 2 diabetes
# Test Results: Fasting glucose 145 mg/dL; HbA1c 7.8%
# Physician Query: What is the most likely diagnosis, and what outpatient treatment plan should be followed?
# """)

# if st.button("Generate Diagnosis & Plan"):
#     with st.spinner("Analyzing..."):
#         result = clinical_qa_chain.run({"query": query_text})
#         st.markdown("### 💡 Clinical Decision Support Output")
#         st.markdown(result)


# ------------------------------
# 0. MUST be first Streamlit command
# ------------------------------
import streamlit as st
st.set_page_config(page_title="🩺 MediMind Clinical Assistant")

# ------------------------------
# 1. Imports and environment
# ------------------------------
from dotenv import load_dotenv
import os
from datetime import datetime
import pytz

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found. Please add it to your .env file.")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

# ------------------------------
# 2. Load FAISS vector store
# ------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local("faiss_index_1", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# ------------------------------
# 3. Prompt template for RAG
# ------------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a clinical decision support assistant. Use the retrieved context and outpatient guidelines (ADA, NICE, WHO) to provide a comprehensive diagnosis and treatment plan.

Context:
{context}

Question:
{question}

Instructions:
- ONLY recommend outpatient-safe diagnosis and treatment. If any red flags are detected, mention them clearly.
- Follow evidence-based outpatient guidelines.
- Fill in EVERY section below with complete, detailed information.
- DO NOT skip any section.

Format your response like this:

**1. Symptom Analysis**
- Differential Diagnoses (include % likelihood + 1-line reasoning per item)
- Red Flags (list if any, else write "None")

**2. Clinical Decision Support**
- Immediate Outpatient Actions:
    • Vitals to check
    • Labs to order (e.g., HbA1c, CBC, TSH, etc.)
    • Physical assessments
- Medications:
    • Name
    • Dose
    • Frequency
    • Purpose
- Lifestyle Modifications:
    • Diet changes (e.g., low glycemic index, reduced carbs)
    • Exercise (type, intensity, duration per week)
    • Smoking cessation if applicable
- Follow-Up Plan:
    • When to repeat labs
    • When to schedule next visit
    • Any specialist referrals needed
- Documentation Notes:
    • Summary of findings
    • Diagnosis and plan in 1-2 sentences

Be concise but complete. Structure clearly. Do not leave any field blank.
"""
)

# ------------------------------
# 4. Initialize LLM & QA Chain
# ------------------------------
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
clinical_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False
)

# ------------------------------
# 5. Memory setup
# ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ------------------------------
# 6. Streamlit UI
# ------------------------------
st.title("🩺 MediMind: Outpatient Clinical Assistant")

# Sidebar chat history
st.sidebar.title("📜 Chat History")
for entry in reversed(st.session_state.chat_history):
    st.sidebar.markdown(f"**{entry['timestamp']}**")
    st.sidebar.markdown(f"> {entry['user_input']}")
    st.sidebar.markdown(f"{entry['response']}")

# Input form
with st.form(key="query_form"):
    st.subheader("📥 Enter Clinical Information")
    context = st.text_area("Context", placeholder="Example: A 6-year-old boy presents with fever and rash in a rural clinic.")
    symptoms = st.text_area("Symptoms", placeholder="Example: Fever for 4 days, non-itchy rash, sore throat, headache")
    history = st.text_area("Medical History", placeholder="Example: No chronic conditions, up to date on vaccinations")
    test_results = st.text_area("Test Results", placeholder="Example: Negative rapid strep test, WBC count 12,000/µL")
    question = st.text_area("Physician Query", placeholder="Example: What could this be? Should I refer to a hospital?")
    submit = st.form_submit_button("🩺 Get Clinical Guidance")

if submit:
    full_query = f"""
Context: {context}
Symptoms: {symptoms}
Medical History: {history}
Test Results: {test_results}
Physician Query: {question}
"""
    with st.spinner("🧠 Thinking..."):
        result = clinical_qa_chain.run({"query": full_query})

    # Display the result
    st.markdown("### 💡 Clinical Decision Support Output")
    st.markdown(result)

    # Save chat history with timestamp
    timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M")
    st.session_state.chat_history.append({
        "timestamp": timestamp,
        "user_input": full_query,
        "response": result
    })
