import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_graph_context(paper_id, G, top_n=3):
    if paper_id not in G: return []
    neighbors = list(G.neighbors(paper_id))
    neighbors = list(set(neighbors))
    return neighbors[:top_n]

class GraphSearchEngine:
    def __init__(self, db_file='graph_vector.pkl', kg_file='knowledge_graph.pkl'):
        # Thư mục gốc của dự án
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Kiểm tra file pkl ưu tiên
        self.db_path = os.path.join(root_dir, db_file) if not os.path.exists(db_file) else db_file
        self.kg_path = os.path.join(root_dir, kg_file) if not os.path.exists(kg_file) else kg_file
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Missing database at {self.db_path}. Please run specter/database_vector.py.")
            
        print(f"--- Loading Binary Module: {self.db_path} ---")
        # SỬA LỖI TẠI ĐÂY: Dùng read_pickle thay vì read_csv
        self.df_db = pd.read_pickle(self.db_path)
        
        # Chuyển đổi sang matrix numpy để tính similarity nhanh
        self.matrix_graph = np.stack(self.df_db['graph_vector'].values)
        
        self.G = None
        if os.path.exists(self.kg_path):
            with open(self.kg_path, 'rb') as f:
                self.G = pickle.load(f)
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def search(self, query, top_k=10, expand_n=3):
        query_vector = self.encoder.encode([query])
        similarity_scores = cosine_similarity(query_vector, self.matrix_graph)[0]
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            paper_id = self.df_db['paper_id'].iloc[idx]
            title = self.df_db['title'].iloc[idx]
            score = similarity_scores[idx]
            neighbors = []
            if self.G and len(results) < 1:
                neighbor_ids = get_graph_context(paper_id, self.G, top_n=expand_n)
                for nid in neighbor_ids:
                    match = self.df_db[self.df_db['paper_id'] == nid]
                    if not match.empty:
                        neighbors.append({"paper_id": nid, "title": match['title'].values[0]})
            
            results.append({"paper_id": paper_id, "title": title, "score": float(score), "neighbors": neighbors})
        return results
