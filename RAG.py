from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = ["doc1 text", "doc2 text", "doc3 text", "cat", "caterpillar"]
embeddings = model.encode(docs)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def vector_search(query, k=5):
    q_emb = model.encode([query]) # makes it a batch of size 1
    print(f"these are the q_embs: {q_emb}")
    D, I = index.search(np.array(q_emb), k)
    print([D, I])
    return [docs[i] for i in I[0]]

RES = vector_search("cat", 2)
print(RES)
