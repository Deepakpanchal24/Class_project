import os
import gradio as gr
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


#Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# One-time setup (run once when app starts)
def setup_rag_chain():
    # URLs to crawl
    urls = [
        'https://my.clevelandclinic.org/health/diseases/7104-diabetes#overview',
    'https://my.clevelandclinic.org/health/diseases/7104-diabetes#symptoms-and-causes',
    'https://my.clevelandclinic.org/health/diseases/7104-diabetes#diagnosis-and-tests',
    'https://my.clevelandclinic.org/health/diseases/7104-diabetes#management-and-treatment',
    'https://my.clevelandclinic.org/health/diseases/7104-diabetes#prevention',
    'https://www.medicalnewstoday.com/articles/communicable-diseases#types-and-symptoms',
    'https://www.medicalnewstoday.com/articles/communicable-diseases#Common-communicable-diseases',
    'https://www.medicalnewstoday.com/articles/communicable-diseases#causes',
    'https://www.medicalnewstoday.com/articles/communicable-diseases#treatment',
    'https://my.clevelandclinic.org/health/diseases/17724-infectious-diseases#what-are-infectious-diseases', 
    'https://my.clevelandclinic.org/health/diseases/17724-infectious-diseases#management-and-treatment',
     'https://www.tataaig.com/knowledge-center/health-insurance/list-of-communicable-diseases'
    ]

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 2})

    # Prompt format
    system_prompt = (
        """You are a clinical decision support assistant.

Given the context from a medical textbook and the details of a patient's case, analyze the symptoms and suggest likely diagnoses with reasoning and urgency.

Respond in this format:

**Context**: <summary of the case in 1 sentence>  
**Input**:
- Symptoms: ...
- Medical History: ...
- Test Results: ...
- Physician Query: ...

**Output (Symptom Analysis)**:
- Differential Diagnoses:
  - <Diagnosis 1>: <probability>% – <reasoning>
  - <Diagnosis 2>: ...
- Alert: <Any urgent action or referral>"""
    )

    template = system_prompt + """

Context:
{context}

Physician Query:
{input}
"""

    prompt = PromptTemplate.from_template(template)
    llm = OpenAI(temperature=0.4, max_tokens=500)
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain

# Load the RAG chain (on startup)
rag_chain = setup_rag_chain()

# Inference function
def clinical_diagnosis(query):
    try:
        response = rag_chain.invoke({"input": query})
        return response.get('answer', 'No answer returned.')
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
interface = gr.Interface(
    fn=clinical_diagnosis,
    inputs=gr.Textbox(lines=6, placeholder="Enter the clinical case or symptoms..."),
    outputs=gr.Textbox(lines=15),
    title="MediMind: Healthcare Assistance",
    description="Enter patient case details to receive AI-assisted differential diagnosis based on trusted medical knowledge."
)

# Launch app
interface.launch()
