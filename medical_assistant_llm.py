import streamlit as st
import pandas as pd
import os
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = Ollama(model="llama3")

# Prompt template
template = """
You are MediMind, an expert AI assistant for doctors. Analyze the following patient case and output:

1. Symptom Analysis (Differential Diagnoses with % likelihood)
2. Clinical Decision Support (immediate actions, medications, follow-ups)

Case Details:
- Symptoms: {symptoms}
- Medical History: {history}
- Test Results: {tests}
- Physician Query: {query}

Output:
"""

prompt = PromptTemplate(
    input_variables=["symptoms", "history", "tests", "query"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.title("MediMind - Clinical AI Assistant")
st.markdown("Provide patient case details below:")

with st.form("case_form"):
    # Patient details
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    address = st.text_input("Address")
    mobile = st.text_input("Mobile Number")

    # Medical case details
    symptoms = st.text_area("Symptoms", placeholder="e.g. Chest pain, shortness of breath...")
    history = st.text_area("Medical History", placeholder="e.g. Hypertension, diabetes...")
    tests = st.text_area("Test Results", placeholder="e.g. ECG abnormal, blood glucose elevated...")
    query = st.text_area("Physician Query", placeholder="e.g. What’s the diagnosis and next steps?")

    submitted = st.form_submit_button("🔍 Analyze")

if submitted:
    with st.spinner("Analyzing patient case..."):
        result = chain.run({
            "symptoms": symptoms,
            "history": history,
            "tests": tests,
            "query": query
        })

    st.subheader("MediMind Response")
    st.write(result)

    # Prepare data to save
    data = {
        "Name": name,
        "Age": age,
        "Gender": gender,
        "Address": address,
        "Mobile": mobile,
        "Symptoms": symptoms,
        "Medical History": history,
        "Test Results": tests,
        "Physician Query": query,
        "MediMind Response": result
    }

    df = pd.DataFrame([data])

    # Save or append to CSV
    csv_file = "patient_cases.csv"
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)

    st.success("✅ Case details and analysis saved to 'patient_cases.csv'")
