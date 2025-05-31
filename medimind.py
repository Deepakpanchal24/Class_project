

#------------------------------Medimind Project------------------------------



# -------------------------
# 📦 Imports
# -------------------------
import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv



# -------------------------
# 🔐 Load API Key 
# -------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI()



# -------------------------
# 🔧 Streamlit Config
# -------------------------
st.set_page_config(page_title="MediMind - Healthcare AI", layout="wide")



# -------------------------
# 🧠 Prompt Template Builder
# -------------------------
def build_prompt(symptoms, history, tests, query):
    return f"""
You are a clinical decision support AI system called MediMind.

Given the patient information below, provide:
1. Differential diagnoses with probabilities.
2. Immediate clinical actions recommended, including:
   - Medication name
   - Drug class
   - Dosage
   - Frequency
   - Duration
3. Follow-up recommendations.
4. A structured EMR-style documentation summary.

Patient Info:
- Symptoms: {symptoms}
- Medical History: {history}
- Test Results: {tests}
- Physician Query: {query}

Be concise, medically accurate, and provide medication suggestions based on standard clinical guidelines.
"""



# -------------------------
# 🤖 Query OpenAI GPT-4
# -------------------------
def query_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()



# -------------------------
# 💻 Streamlit UI
# -------------------------
st.title("🧠 MediMind: AI for Symptom Analysis & Clinical Decision Support")

# Manual Inputs
patient_id = st.text_input("🆔 Patient ID", "P001")
patient_name = st.text_input("👤 Patient Name", "John Doe")
age = st.text_input("🎂 Patient Age", "52")
gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"])
symptoms = st.text_area("🩺 Symptoms", placeholder="e.g., Chest pain for 2 hours, radiating to left arm...")
history = st.text_area("📜 Medical History", placeholder="e.g., Hypertension, smoker...")
tests = st.text_area("🧪 Test Results", placeholder="e.g., ECG shows ST-elevation; troponin I: 2.5 ng/mL")
query = st.text_area("👨‍⚕️ Physician's Query", "What’s the likely diagnosis, and what should we do next?")



# TB-specific predefined clinical actions
tb_custom_actions = """
### 🔒 Predefined Clinical Actions for Tuberculosis:

- Isolate the patient to prevent potential spread of TB.
- Initiate empiric anti-TB therapy considering the high likelihood of TB.
- Send sputum for Acid-Fast Bacilli (AFB) smear and culture.
- Start prophylactic treatment for PCP with trimethoprim-sulfamethoxazole due to HIV status.
- Consult infectious disease specialist.
- Follow-up Recommendations:
"""

# Run LLM
if st.button("🚀 Run MediMind Analysis"):
    prompt = build_prompt(symptoms, history, tests, query)
    with st.spinner("MediMind is thinking..."):
        output = query_llm(prompt)

    st.success("✅ Analysis Complete")
    st.markdown("### 🧾 MediMind Clinical Decision Support Response")

    # Check for TB for predefined actions
    if "tuberculosis" in symptoms.lower() or "tuberculosis" in history.lower():
        st.markdown(tb_custom_actions)

    st.write(output)

    # Save to CSV
    result_row = {
        "Patient ID": patient_id,
        "Patient Name": patient_name,
        "Age": age,
        "Gender": gender,
        "Symptoms": symptoms,
        "Medical History": history,
        "Test Results": tests,
        "Physician Query": query,
        "AI Output": output
    }

    if not os.path.exists("medimind_saved_results.csv"):
        pd.DataFrame([result_row]).to_csv("medimind_saved_results.csv", index=False)
    else:
        pd.DataFrame([result_row]).to_csv("medimind_saved_results.csv", mode='a', header=False, index=False)



# -------------------------
# 📊 Display Saved Results
# -------------------------
# Display Saved Results (safe mode)
if os.path.exists("medimind_saved_results.csv"):
    st.markdown("### 📂 Previously Saved Results")
    try:
        saved_df = pd.read_csv("medimind_saved_results.csv", on_bad_lines='skip')  # safe read
        st.dataframe(saved_df, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error loading saved results: {e}")