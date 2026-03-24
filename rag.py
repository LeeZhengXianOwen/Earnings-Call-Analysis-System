import os
from dotenv import load_dotenv
import faiss
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq

# Token limit problems with current LLM - will explore other options

def retrieve(query, model, index, chunks, top_k = 5, filter_company = None, filter_quarter = None):
    # Embed the query
    q_emb = model.encode([query], convert_to_numpy = True)
    faiss.normalize_L2(q_emb)

    # Search — over-fetch to allow for filtering
    scores, indices = index.search(q_emb, top_k * 3)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = chunks.iloc[idx]

        # Optional filters
        if filter_company and row["company"].lower() != filter_company.lower():
            continue
        if filter_quarter and row["quarter"] != filter_quarter:
            continue

        results.append({
            "chunk_id": row["chunk_id"],
            "company": row["company"],
            "quarter": row["quarter"],
            "score": float(score),
            "chunk_text": row["chunk_text"]
        })

        if len(results) == top_k:
            break

    return results

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join([
        f"[{chunk['company']}]\n{chunk['chunk_text']}"
        for chunk in retrieved_chunks
    ])
    prompt = (
        f"{query}\n\n"
        f"Use ONLY the information from the exercpts provided below to answer the question.\n\n"
        f"Relevant excerpts from earnings call transcripts:\n{context}"
    )
    return prompt

def generate(prompt, client):
    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.0,
        max_tokens = 512,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Load global parameters (models, data etc.)
    load_dotenv()
    BASE_DIR = Path(__file__).parent
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(str(BASE_DIR / "Results" / "retriever_data" / "faiss.index"))
    chunks = pd.read_csv(str(BASE_DIR / "Results" / "chunk_data" / "rag_chunks.csv"))
    client = Groq(api_key = os.getenv("GROQ_API_KEY")) # Your LLM/API key here

    print("Type 'quit' to exit\n")

    while True:
        query = input("Your question: ").strip()

        if query.lower() == "quit":
            print("Goodbye!")
            break
        if not query:
            continue

        # Placeholder line to fish out company and quarter for retrieve function parameter
        # Subject to change in retrieval

        retrieved_chunks = retrieve(query, model, index, chunks, top_k = 5)
        prompt = build_prompt(query, retrieved_chunks)
        answer = generate(prompt, client)
        
        print(f"\nAnswer: {answer}\n")
        print(f"Retrieved chunks: \n")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"  {i}. {chunk['company']} {chunk['quarter']} - {chunk['chunk_text'][:60]}... {chunk['score']}\n")