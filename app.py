from Class_RAG_graph_vector import GraphRAG
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Sample corpus (replace later with DB)
# -----------------------------
docs = [
    "Cats are small domesticated animals",
    "Dogs are loyal pets",
    "Wolves are ancestors of dogs",
    "Quantum physics studies particles",
    "Particles interact through forces"
]

# -----------------------------
# Initialize GraphRAG once (important!)
# -----------------------------
rag = GraphRAG(docs)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="GraphRAG Service")


# -----------------------------
# Request schema
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    k: int = 2
    hops: int = 1


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Graph-only search
# -----------------------------
@app.post("/graph-search")
def graph_search(req: QueryRequest):
    results = rag.graph_search(req.query, k=req.k, hops=req.hops)
    return {
        "query": req.query,
        "mode": "graph",
        "results": results
    }


# -----------------------------
# Hybrid search
# -----------------------------
@app.post("/hybrid-search")
def hybrid_search(req: QueryRequest):
    results = rag.hybrid_search(req.query, k=req.k, hops=req.hops)
    return {
        "query": req.query,
        "mode": "hybrid",
        "results": results
    }

@app.get("/")
def root():
    return {
        "message": "GraphRAG service is running",
        "endpoints": [
            "/graph-search",
            "/hybrid-search",
            "/health"
        ]
    }
