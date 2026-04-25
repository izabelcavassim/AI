
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# Sample corpus
docs = [
    "Cats are small domesticated animals",
    "Dogs are loyal pets",
    "Wolves are ancestors of dogs",
    "Quantum physics studies particles",
    "Particles interact through forces"
]

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(docs)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Build vector search
def vector_search(query, k=2):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return list(I[0]), list(D[0])

def build_knn_graph(embeddings, k=3):
    G = nx.Graph()

    sim = cosine_similarity(embeddings)

    for i in range(len(embeddings)):
        neighbors = np.argsort(sim[i])[-k-1:-1]  # top-k excluding self
        for j in neighbors:
            G.add_edge(i, j, weight=sim[i][j])

    return G

# Mapping query to the graoh entry point
def get_seed_nodes(query, k=2):
    q_emb = model.encode([query])
    sim = cosine_similarity(q_emb, embeddings)[0]

    top_k = np.argsort(sim)[-k:]
    return list(top_k)

# Graph expansion
def expand_graph(G, seed_nodes, hops=1):
    visited = set(seed_nodes)

    for node in seed_nodes:
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=hops)
        visited.update(neighbors.keys())

    return list(visited)

# Ranking nodes, semantic and structure
def rank_nodes(query, nodes):
    q_emb = model.encode([query])
    node_embs = embeddings[nodes]

    semantic_scores = cosine_similarity(q_emb, node_embs)[0]

    return sorted(zip(nodes, semantic_scores), key=lambda x: x[1], reverse=True)

# RAG search
def graph_rag_search(query, embeddings, k=2, hops=1):

    G = build_knn_graph(embeddings)
    seeds = get_seed_nodes(query, k=k)
    expanded = expand_graph(G, seeds, hops=hops)
    ranked = rank_nodes(query, expanded)

    return [docs[i] for i, _ in ranked]


# Hybrid ranking
def hybrid_rank(query, nodes, G, alpha=0.7):
    q_emb = model.encode([query])
    node_embs = embeddings[nodes]

    semantic_scores = cosine_similarity(q_emb, node_embs)[0]

    scores = []
    for i, node in enumerate(nodes):
        # graph score = degree (simple proxy)
        graph_score = len(list(G.neighbors(node)))

        score = alpha * semantic_scores[i] + (1 - alpha) * graph_score
        scores.append((node, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)


# Hybrid search: vector + graph scoring
def hybrid_search(query, embeddings, G, k=2, hops=1):
    # Step 1: vector search
    seed_nodes, distances = vector_search(query, k=k)

    #Step 2: graph expansion
    expanded_nodes = expand_graph(G, seed_nodes, hops=hops)

    #Step 3: re-rank
    ranked = hybrid_rank(query, expanded_nodes, G)

    return [docs[i] for i, _ in ranked]
     
G = build_knn_graph(embeddings)
print(f"This is graph rag search: {graph_rag_search("animal related to dogs", embeddings)}\n")
print(f"This is hybrid rag search: {hybrid_search("animal related to dogs", embeddings, G)}")
