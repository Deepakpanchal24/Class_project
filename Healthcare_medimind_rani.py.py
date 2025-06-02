
# Import Required Libraries

import streamlit as st
import pandas as pd
import groq
import os
from dotenv import load_dotenv

# Load API Key from .env File


# Use .env file to securely store your GROQ API key.
# You should create a .env file with the line: GROQ_API_KEY=your_key_here

load_dotenv()
groq.api_key = os.getenv("GROQ_API_KEY")

# Ensure the API key is set

# Load Medical Dataset (CSV)

@st.cache_data
def load_dataset():
    """
    Load the synthetic medical data from the uploaded CSV file.
    Caching is enabled for performance.
    """
    try:
        df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\PROJECT_HEALTHCARE\\synthetic_medical_data_genai.csv")

        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please make sure 'synthetic_medical_data_genai.csv' is in the same directory.")
        return pd.DataFrame()


# Construct the Prompt

def create_prompt(name, gender, age, symptoms, history, tests, question):
    """
    Build a prompt in a structured format that includes:
    - Name
    - Gender
    - Age
    - Symptoms
    - Medical history
    - Test results
    - A physician's query
    """
    prompt = f"""
You are MediMind, a helpful and knowledgeable AI assistant for clinical diagnosis and treatment planning.

You will be given structured clinical data about a patient case, and you must:
1. Analyze the symptoms and test results.
2. Provide a list of possible differential diagnoses with probabilities.
3. Give appropriate clinical decision support including:
   - Immediate actions required
   - Recommended treatment plan
   - Suggested follow-up investigations or monitoring

PATIENT CASE DETAILS:
- Name: {name}
- Gender: {gender}
- Age: {age}
- Symptoms: {symptoms}
- Medical History: {history}
- Test Results: {tests}

PHYSICIAN QUERY:
"{question}"

Please respond in this format:

--- DIAGNOSIS ---
[List differential diagnoses and their likelihoods]

--- CLINICAL DECISION SUPPORT ---
- Immediate Actions:
- Treatment Plan:
- Follow-Up Recommendations:
"""
    return prompt


# Call Groq's 


import groq

# Initialize client only once globally
client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

def get_medimind_response(prompt):
    """
    Uses GROQ's OpenAI-compatible Chat API to generate a clinical response.
    """
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful and accurate AI assistant for doctors."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"


#streamlit App UI

# Set up the web app page
st.set_page_config(page_title="MediMind - AI for Clinical Support", layout="wide")

# Title
st.title("MediMind: AI-Powered Clinical Decision Support System")

# Subtitle
st.markdown("This application uses **GROQ's GPT model** to assist healthcare professionals in analyzing symptoms, providing differential diagnoses, and suggesting treatment plans.")


# 🧾 Input Fields
st.header("🔍 Enter Patient Case Information")

# Text areas for structured clinical input
name =st.text_input("Enter your name")
gender = st.selectbox("Select gender", ["Select", "Male", "Female", "Other"])
age = st.number_input("Enter you age", min_value=0, max_value=120, step=1)

symptoms = st.text_area("Symptoms", placeholder="E.g., chest pain for 2 hours, shortness of breath", height=100)
history = st.text_area("Medical History", placeholder="E.g., diabetes, hypertension, smoker", height=100)
tests = st.text_area("Test Results", placeholder="E.g., ECG shows ST-elevation, troponin I = 2.5 ng/mL", height=100)
query = st.text_input("Physician Query", placeholder="E.g., What’s the likely diagnosis and treatment plan?")

# Submit button
if st.button("Run Clinical Analysis with MediMind"):
    
    if not all([name, gender != "Select", age, symptoms, history, tests, query]):
        st.warning("Please fill out all fields before submitting.")
    else:
        with st.spinner(" MediMind is analyzing the case..."):
            # Generate prompt
            final_prompt = create_prompt(name, gender, age, symptoms, history, tests, query)

            # Get Groq response
            output = get_medimind_response(final_prompt)

        # Display output
        st.subheader("MediMind’s Response")
        st.markdown(output)


#  Display Sample Dataset
st.markdown("---")
st.subheader("Sample Reference Cases from Dataset")
df = load_dataset()
if not df.empty:
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.info("No reference data loaded.")
