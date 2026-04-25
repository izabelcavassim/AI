from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class GraphRAG:
    def __init__(self, docs, model_name="all-MiniLM-L6-v2", k_graph=3):
        self.docs = docs
        self.k_graph = k_graph

        # embeddings
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(docs)

        # FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

        # graph
        self.graph = self._build_knn_graph()

    # -----------------------------
    # GRAPH CONSTRUCTION
    # -----------------------------
    def _build_knn_graph(self):
        G = nx.Graph()

        sim = cosine_similarity(self.embeddings)

        for i in range(len(self.docs)):
            G.add_node(i)

            neighbors = np.argsort(sim[i])[-self.k_graph - 1:-1]

            for j in neighbors:
                G.add_edge(i, j, weight=sim[i][j])

        return G

    # -----------------------------
    # VECTOR SEARCH (FAISS)
    # -----------------------------
    def vector_search(self, query, k=2):
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb), k)
        return list(I[0]), list(D[0])

    # -----------------------------
    # GRAPH EXPANSION
    # -----------------------------
    def expand_graph(self, seed_nodes, hops=1):
        visited = set(seed_nodes)

        for node in seed_nodes:
            if node not in self.graph:
                continue

            neighbors = nx.single_source_shortest_path_length(
                self.graph, node, cutoff=hops
            )
            visited.update(neighbors.keys())

        return list(visited)

    # -----------------------------
    # RERANKING
    # -----------------------------
    def rerank(self, query, nodes, alpha=0.7):
        q_emb = self.model.encode([query])
        node_embs = self.embeddings[nodes]

        semantic_scores = cosine_similarity(q_emb, node_embs)[0]

        scores = []
        for i, node in enumerate(nodes):
            graph_score = len(list(self.graph.neighbors(node)))
            score = alpha * semantic_scores[i] + (1 - alpha) * graph_score
            scores.append((node, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    # -----------------------------
    # PURE GRAPH RAG
    # -----------------------------
    def graph_search(self, query, k=2, hops=1):
        seeds, _ = self.vector_search(query, k=k)
        expanded = self.expand_graph(seeds, hops=hops)
        ranked = self.rerank(query, expanded)

        return [self.docs[i] for i, _ in ranked]

    # -----------------------------
    # HYBRID SEARCH (ENTRY POINT + GRAPH)
    # -----------------------------
    def hybrid_search(self, query, k=2, hops=1):
        seeds, _ = self.vector_search(query, k=k)
        expanded = self.expand_graph(seeds, hops=hops)
        ranked = self.rerank(query, expanded)

        return [self.docs[i] for i, _ in ranked]
