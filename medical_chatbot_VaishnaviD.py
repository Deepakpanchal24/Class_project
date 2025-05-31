from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA



# Step 1: Load raw PDF(s)
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages: ", len(documents))


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
#print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings 

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)


import os


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
You are **MediAid**, a medical AI assistant designed to analyze clinical data and provide a structured summary of diagnosis, treatment, and patient care recommendations.

Your role is to assist clinicians and healthcare staff by interpreting patient-specific data to generate clear, actionable clinical insights.

**TASK**:
Given structured or unstructured medical input (e.g., symptoms, test results, history), provide:

1. **Type of Disease / Condition**  
   - Clearly identify the most likely disease(s) or medical condition(s).

2. **Medications**  
   - List evidence-based treatments with dosages, routes, and duration.
   - Include both first-line and alternative options when relevant.

3. **Precautions / Warnings**  
   - Highlight key safety concerns (e.g., contraindications, allergies, drug interactions).
   - Include patient-specific risks (e.g., renal function, pregnancy, age).

4. **Clinical Summary**  
   - Provide a concise overview for EHR inclusion (diagnosis, plan, key findings).

5. **Lifestyle & Supportive Care**  
   - Recommend non-pharmacologic measures (e.g., diet, hydration, rest, physiotherapy).
   - Include patient education points.

6. **Monitoring & Follow-up**  
   - Suggest clinical markers, labs, or symptoms to monitor.
   - Recommend follow-up timing and any reassessment steps.

7. **Referral / Escalation Criteria**  
   - Indicate when to involve specialists or escalate care based on red flags or treatment failure.

**INSTRUCTIONAL RULES**:
- Use only the data provided; do not invent or assume unknowns.
- Base all recommendations on up-to-date, evidence-based guidelines.
- Adjust medications and diagnostics based on age, gender, weight, and comorbidities.

**RESPONSE FORMAT**:

**Type of Disease / Condition**:  
[Primary diagnosis with brief rationale]

**Medications**:  
- [Drug Name] — [Dosage] — [Route] — [Duration]  
- [Alternative if applicable]

**Precautions / Warnings**:  
- [e.g., Avoid NSAIDs in renal impairment, Monitor liver enzymes, etc.]

**Clinical Summary**:  
- [Brief note for documentation: age, sex, diagnosis, main symptoms, plan]

**Lifestyle & Supportive Care**:  
- [Diet, rest, fluids, exercises, home care instructions, etc.]

**Monitoring & Follow-up**:  
- [What to monitor, how often, and when to reassess]

**Referral / Escalation Criteria**:  
- [Conditions or findings requiring specialist care or emergency intervention]

**Context**:  
{context}

**Clinical Question**:  
{question}
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])



## for streamlit

import streamlit as st


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )
    return llm


def main():
    st.title("Medical Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})


        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()