import ollama
from retrieval import retrieve_relevant_chunks

def generate_answer(query):
    retrieved_docs = retrieve_relevant_chunks(query)

    context = "\n".join(retrieved_docs)
    
    prompt = f"Use the following information to answer:\n{context}\n\nQuestion: {query}"
    
    response = ollama.chat(model="deepseek-r1:8b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
