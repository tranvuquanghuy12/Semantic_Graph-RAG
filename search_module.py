import pandas as pd
import numpy as np
import pickle
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os

def parse_vector_from_csv(df_column):
    """Giải mã vector từ string trong CSV"""
    parsed_lists = df_column.apply(ast.literal_eval)
    return np.array(parsed_lists.tolist())

def get_graph_context(paper_id, G, top_n=3):
    """Truy xuất các bài báo liên quan (neighbor) từ Knowledge Graph"""
    if paper_id not in G:
        return []
    
    # Lấy các bài báo mà bài này trích dẫn (out_edges) hoặc được trích dẫn (in_edges)
    neighbors = list(G.neighbors(paper_id))
    # Nếu là đồ thị có hướng, ta lấy cả người trích dẫn nó
    if hasattr(G, 'predecessors'):
        neighbors += list(G.predecessors(paper_id))
        
    # Loại bỏ trùng lặp
    neighbors = list(set(neighbors))
    return neighbors[:top_n]

def Graph_RAG_Search():
    print("=== HỆ THỐNG TÌM KIẾM HYBRID (GAT + GRAPH-RAG) ===")
    
    # 1. TẢI DATABASE VÀ KNOWLEDGE GRAPH
    db_file = 'graph_vector.csv'
    kg_file = 'knowledge_graph.pkl'
    
    if not os.path.exists(db_file):
        print(f"[ERROR] Thiếu file {db_file}. Vui lòng chạy database_vector.py trước.")
        return
        
    df_db = pd.read_csv(db_file)
    matrix_base_text = parse_vector_from_csv(df_db['base_text_vector'])
    
    G = None
    if os.path.exists(kg_file):
        with open(kg_file, 'rb') as f:
            G = pickle.load(f)
        print(f"[OK] Đã nạp Knowledge Graph. Quy quy mô: {G.number_of_nodes()} nodes.")
    else:
        print("[WARNING] Không tìm thấy Knowledge Graph. Hệ thống sẽ chỉ dùng Vector Search.")

    # 2. KHỞI TẠO MODEL
    text_model = SentenceTransformer('all-MiniLM-L6-v2') 

    # 3. TRUY VẤN
    query = "application of graph models in network citation and deep learning"
    print(f"\nQUERY: '{query}'")
    
    # Lấy Vector GAT thay vì Vector thô
    matrix_graph = parse_vector_from_csv(df_db['graph_vector'])
    
    # Bước 1: Dense Retrieval sử dụng GAT-refined vector! (Ăn tiền nốt ở đây)
    query_vector = text_model.encode([query])
    similarity_scores = cosine_similarity(query_vector, matrix_graph)[0]
    top_1_idx = similarity_scores.argmax()
    
    top_paper_id = df_db['paper_id'].iloc[top_1_idx]
    top_title = df_db['title'].iloc[top_1_idx]
    
    print(f"\n[PHASE 1: DENSE SEARCH]")
    print(f"-> Bài báo gốc tìm thấy: {top_title} (ID: {top_paper_id})")

    # Bước 2: Graph Traversal (Knowledge Graph Expansion)
    print(f"\n[PHASE 2: GRAPH-RAG EXPANSION]")
    if G:
        neighbor_ids = get_graph_context(top_paper_id, G)
        if neighbor_ids:
            print(f"-> Tìm thấy {len(neighbor_ids)} bài báo liên quan trong KG (Citation neighbors):")
            for nid in neighbor_ids:
                # Tìm tiêu đề trong DB
                match = df_db[df_db['paper_id'] == nid]
                if not match.empty:
                    print(f"   - {match['title'].values[0]}")
                else:
                    print(f"   - Paper ID: {nid} (Thông tin nằm ngoài tập dữ liệu hiện tại)")
        else:
            print("-> Không tìm thấy node hàng xóm cho bài báo này trong KG.")
    else:
        print("-> Bỏ qua bước mở rộng đồ thị.")

    # Bước 3: RAG Prompting (Lý thuyết)
    print(f"\n[PHASE 3: GENERATING RAG PROMPT]")
    print("Mẫu Context gửi cho LLM:")
    print(f"--- CONTEXT START ---")
    print(f"Main Paper: {top_title}")
    # (Thêm abstract và neighbors vào đây)
    print(f"--- CONTEXT END ---")
    print(f"Prompt: 'Dựa vào bài báo chính và các bài báo liên quan trong đồ thị trích dẫn nêu trên, hãy trả lời: {query}'")

if __name__ == "__main__":
    Graph_RAG_Search()
