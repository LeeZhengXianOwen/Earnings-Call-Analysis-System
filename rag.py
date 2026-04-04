import os
from dotenv import load_dotenv
import faiss
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# Token limit problems with current LLM - will explore other options

def extract_filters(query):
    """
    Basic function to extract company and quarter (or other metadata filters)
    from the user query, to aid in the retrieval process.
    """
    company = []
    quarter = []

    for comp in companies:
        if comp.lower() in query.lower() and comp.lower() not in company:
            company.append(comp)

    for q in quarters:
        if q.lower() in query.lower() and q.lower() not in quarter:
            quarter.append(q)

    if not company:
        company = None
    if not quarter:
        quarter = None
    return company, quarter

from qdrant_client.models import Filter, FieldCondition, MatchAny

def retrieve(query, model, db, top_k=5, filter_company=None, filter_quarter=None):
    # Embed the query
    q_emb = model.encode([query], convert_to_numpy=True)[0].tolist()

    # Build metadata filter if provided
    conditions = []
    if filter_company:
        companies = filter_company if isinstance(filter_company, list) else [filter_company]
        conditions.append(FieldCondition(key="company", match=MatchAny(any=companies)))
    if filter_quarter:
        quarters = filter_quarter if isinstance(filter_quarter, list) else [filter_quarter]
        conditions.append(FieldCondition(key="quarter", match=MatchAny(any=quarters)))

    search_filter = Filter(must=conditions) if conditions else None

    # Search — filter applied inside Qdrant, no over-fetching needed
    results = db.query_points(
        collection_name="chunks",
        query=q_emb,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True
    )

    return [
        {
            "chunk_id":   r.payload["chunk_id"],
            "company":    r.payload["company"],
            "quarter":    r.payload["quarter"],
            "score":      r.score,
            "chunk_text": r.payload["chunk_text"]
        }
        for r in results.points
    ]

def build_prompt(query, retrieved_chunks):
    """
    Builds the augmented prompt using the user query and retrieved chunks
    to be fed into the LLM.
    """
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

def generate(prompt, llm):
    response = llm.chat.completions.create(
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
    db = QdrantClient(path=str(BASE_DIR / "Results" / "retriever_data" / "qdrant_store"))
    llm = Groq(api_key = os.getenv("GROQ_API_KEY")) # Your LLM/API key here
    companies = [c.payload["company"] for c in db.scroll(collection_name="chunks", limit=10000)[0]]
    companies = list(set(companies))
    quarters = [c.payload["quarter"] for c in db.scroll(collection_name="chunks", limit=10000)[0]]
    quarters = list(set(quarters))

    print("Type 'quit' to exit\n")

    while True:
        query = input("Your question: ").strip()

        if query.lower() == "quit":
            print("Goodbye!")
            break
        if not query:
            continue

        filtered_company, filtered_quarter = extract_filters(query)

        retrieved_chunks = retrieve(query, model, db, top_k=5, 
                                    filter_company=filtered_company, filter_quarter=filtered_quarter)
        prompt = build_prompt(query, retrieved_chunks)
        answer = generate(prompt, llm)
        
        print(f"\nAnswer: {answer}\n")
        print(f"Retrieved chunks: \n")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"  {i}. {chunk['company']} {chunk['quarter']} - {chunk['chunk_text'][:60]}... {chunk['score']}\n")