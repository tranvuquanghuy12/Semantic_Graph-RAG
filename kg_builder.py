import json
import networkx as nx
import pickle
import os
import re

def extract_concepts(text, top_n=5):
    """Trích xuất từ khóa đơn giản (Keywords) từ text"""
    # Loại bỏ các từ dừng cơ bản (Stopwords)
    stopwords = {"and", "the", "for", "with", "using", "from", "image", "paper", "data", "results", "based"}
    words = re.findall(r'\b\w{4,}\b', text.lower()) # Lấy từ có 4 ký tự trở lên
    filtered_words = [w for w in words if w not in stopwords]
    
    # Đếm tần suất
    freq = {}
    for w in filtered_words:
        freq[w] = freq.get(w, 0) + 1
    
    # Lấy top N từ xuất hiện nhiều nhất
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_freq[:top_n]]

def build_knowledge_graph(json_path, output_path):
    print(f"--- Đang tải dữ liệu từ {json_path} ---")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.MultiDiGraph() # Dùng MultiDiGraph để hỗ trợ nhiều loại quan hệ

    print("--- Bắt đầu xây dựng Nodes và Edges (Papers & Concepts) ---")
    for paper in data:
        pid = paper['paper_id']
        title = paper.get('title', 'Unknown Title')
        abstract = paper.get('abstract', '')
        
        # 1. Thêm node Paper
        G.add_node(pid, title=title, type='paper', color='#00ccff')
        
        # 2. Thêm và liên kết các Concepts (Thực thể tri thức)
        concepts = extract_concepts(title + " " + abstract, top_n=3)
        for concept in concepts:
            G.add_node(concept, title=concept, type='concept', color='#ff9900')
            G.add_edge(pid, concept, relation='has_concept')
        
        # 3. Thêm các cạnh trích dẫn (CITES)
        citations = paper.get('citations', [])
        for cite in citations:
            target_id = cite.get('paperId')
            if target_id:
                G.add_edge(pid, target_id, relation='cites')

    print(f"Tổng cộng: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")
    
    # Lưu đồ thị
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Đã lưu NEW Knowledge Graph vào: {output_path}")

if __name__ == "__main__":
    # Cập nhật đường dẫn theo cấu trúc phẳng
    DATA_PATH = "papers_clean.json"
    OUTPUT_PATH = "knowledge_graph.pkl"
        
    build_knowledge_graph(DATA_PATH, OUTPUT_PATH)
