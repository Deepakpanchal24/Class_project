
# clinical_genai_app.py

import os
import pandas as pd
import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# -------------------------
# Data Loader
# -------------------------
def load_clinical_data(csv_path):
    df = pd.read_csv(csv_path)
    texts = []
    for _, row in df.iterrows():
        combined = (
            f"Symptoms: {row.get('Symptoms', '')}\n"
            f"Medical History: {row.get('Medical History', '')}\n"
            f"Test Results: {row.get('Test Results', '')}\n"
            f"Physician Query: {row.get('Physician Query', '')}"
        )
        texts.append(combined)
    return texts, df

# -------------------------
# Vector Store Functions
# -------------------------
INDEX_PATH = "faiss_index/index.faiss"
EMBEDDING_DIM = 384
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype('float32')

def create_vector_store(texts):
    embeddings = embed_texts(texts)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"[INFO] Vector store created with {len(texts)} vectors.")

def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Vector store not found. Please create it first.")
    index = faiss.read_index(INDEX_PATH)
    return index

def search_vector_store(index, query, top_k=3):
    query_emb = embed_texts([query])
    distances, indices = index.search(query_emb, top_k)
    return indices[0], distances[0]

# -------------------------
# LLM and RAG Chain
# -------------------------
print("[INFO] Loading HuggingFace text-generation pipeline...")
text_gen_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    device=-1,
    do_sample=True,
    temperature=0.7,
    top_p=0.95
)
print("[INFO] Pipeline loaded.")

def generate_prompt(context_docs, user_query):
    context_text = "\n\n---\n\n".join(context_docs)
    prompt = (
        f"You are a clinical assistant AI. Use the following clinical case information to answer the question.\n\n"
        f"{context_text}\n\n"
        f"Question: {user_query}\n\n"
        f"Answer in a detailed, clear, and clinical manner including symptom analysis, differential diagnosis, treatment plan, and patient summary if applicable."
    )
    return prompt

def query_local_model(prompt):
    max_model_length = 1024
    max_new_tokens = 150
    tokens = text_gen_pipeline.tokenizer.encode(prompt)
    if len(tokens) + max_new_tokens > max_model_length:
        allowed_len = max_model_length - max_new_tokens
        tokens = tokens[-allowed_len:]
        prompt = text_gen_pipeline.tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
        print("[INFO] Prompt truncated to fit model max length.")

    outputs = text_gen_pipeline(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)
    answer = outputs[0]['generated_text'].strip()
    answer = answer[len(prompt):].strip()
    return answer

def get_rag_chain(index, df):
    def rag_query(user_query):
        print("[INFO] Searching vector store for relevant docs...")
        indices, _ = search_vector_store(index, user_query, top_k=3)
        print(f"[INFO] Retrieved indices: {indices}")

        context_docs = []
        for i in indices:
            row = df.iloc[i]
            case_text = (
                f"Disease: {row.get('disease', '')}\n"
                f"Age: {row.get('age', '')}\n"
                f"Gender: {row.get('gender', '')}\n"
                f"Region: {row.get('region', '')}\n"
                f"Symptoms: {row.get('symptoms', '')}\n"
                f"Medical History: {row.get('medical_history', '')}\n"
                f"Test Results: {row.get('test_results', '')}\n"
                f"Differential Diagnoses: {row.get('differential_diagnoses', '')}\n"
                f"Medications: {row.get('medications', '')}\n"
                f"Treatment Plan: {row.get('treatment_plan', '')}\n"
                f"Follow Ups: {row.get('follow_ups', '')}"
            )
            if case_text.strip():
                context_docs.append(case_text)

        if not context_docs:
            context_docs = ["No relevant clinical data found."]

        prompt = generate_prompt(context_docs, user_query)
        print("[INFO] Generated prompt (first 500 chars):")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

        print("[INFO] Querying local model...")
        answer = query_local_model(prompt)
        print("[INFO] Model response generated.")
        return answer

    return rag_query

# -------------------------
# Main App Runner
# -------------------------
if __name__ == "__main__":
    csv_path = r'C:\Users\RAVI\OneDrive\Desktop\chat_doc\medimind_india_raw_data.csv'

    print("[INFO] Loading clinical data...")
    texts, df = load_clinical_data(csv_path)

    if not os.path.exists("faiss_index"):
        print("[INFO] Creating vector store...")
        create_vector_store(texts)
    else:
        print("[INFO] Loading vector store...")

    index = load_vector_store()
    print("[INFO] Vector store loaded.")

    print("[INFO] Getting RAG chain...")
    qa_chain = get_rag_chain(index, df)
    print("[INFO] RAG chain ready.")

    test_query = (
        "A 52-year-old male presents with chest pain radiating to left arm, "
        "ST elevation on ECG, elevated troponin. What is the diagnosis and recommended next steps?"
    )
    print("\n[Terminal] Running test query:")
    print(test_query)

    try:
        answer = qa_chain(test_query)
        print("\n[Terminal] Answer:")
        print(answer)
    except Exception as e:
        print(f"\n[Terminal] Error while querying: {e}")
