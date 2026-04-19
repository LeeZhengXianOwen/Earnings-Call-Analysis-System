import pickle
import string
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi

BASE_DIR   = Path(__file__).parent
CHUNKS_CSV = BASE_DIR / "Results" / "rag_chunks_v2_with_labels.csv"
BM25_PATH  = BASE_DIR / "Results" / "retriever_data" / "bm25_index.pkl"

# Tokeniser — simple whitespace + lowercase + punctuation removal.
# Must be the same function used at query time in rag.py.
def tokenise(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()

# Build index
print(f"Loading chunks from {CHUNKS_CSV} ...")
chunks_df = pd.read_csv(CHUNKS_CSV)
chunks_df = chunks_df.dropna(subset=["chunk_text"]).reset_index(drop=True)
print(f"  {len(chunks_df)} chunks loaded")

tokenised_corpus = [tokenise(text) for text in chunks_df["chunk_text"]]
bm25 = BM25Okapi(tokenised_corpus)

# Save the index and the ordered list of chunk_ids (so we can map
# BM25 result indices back to chunk_ids at query time)
payload = {
    "bm25":      bm25,
    "chunk_ids": chunks_df["chunk_id"].tolist(),
}

BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(BM25_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"BM25 index saved to {BM25_PATH}")