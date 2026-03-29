import pandas as pd
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

def tao_vector_he_thong():
    print("=== HỆ THỐNG TẠO VECTOR ĐA LUỒNG (TEXT + GAT + GRAPH) ===")
    
    file_raw = 'papers_clean.json'
    file_refined = 'refined_embeddings.jsonl' # Đây là file đầu ra của GAT Refinement
    
    if not os.path.exists(file_raw):
        print(f"[ERROR] Không tìm thấy dữ liệu thô '{file_raw}'.")
        return

    # 1. ĐỌC DỮ LIỆU GỐC
    with open(file_raw, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # 2. LUỒNG 1: TEXT EMBEDDING (Bản gốc)
    print("\n[STEP 1] Tạo Base Text Vectors (NanoNet/MiniLM)...")
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    text_data = df['title'] + " " + df['abstract'].fillna('')
    base_vectors = model.encode(text_data.tolist(), show_progress_bar=True)
    df['base_text_vector'] = base_vectors.tolist()

    # 3. LUỒNG 2: GAT REFINED EMBEDDING (Dữ liệu nghiên cứu)
    print("\n[STEP 2] Tích hợp GAT-Refined Vectors...")
    if os.path.exists(file_refined):
        print(f"-> Phát hiện file GAT output: {file_refined}. Đang nạp...")
        refined_data = {}
        with open(file_refined, 'r') as f:
            for line in f:
                d = json.loads(line)
                refined_data[d['paper_id']] = d['embedding']
        
        # Ánh xạ vào DataFrame
        df['graph_vector'] = df['paper_id'].map(refined_data)
        # Điền giá trị mặc định cho những bài không có trong graph (nếu có)
        df['graph_vector'] = df['graph_vector'].apply(lambda x: x if isinstance(x, list) else [0]*768)
        print("[OK] Đã tích hợp thành công GAT Vectors.")
    else:
        print("[WARNING] Không tìm thấy 'refined_embeddings.jsonl'.")
        print("-> Gợi ý: Hãy chạy 'python specter/scripts/gat_refinement.py' trên server trước.")
        df['graph_vector'] = [np.zeros(768).tolist() for _ in range(len(df))]

    # 4. LƯU DATABASE CHO MODULE SEARCH
    print("\n[STEP 3] Xuất Database Vector...")
    file_ket_qua = 'graph_vector.csv'
    # Chỉ lưu các thông tin cần thiết nhất
    cols = ['paper_id', 'title', 'base_text_vector', 'graph_vector']
    df[cols].to_csv(file_ket_qua, index=False)

    
    print("-" * 50)
    print(f"HOÀN THÀNH! File sẵn sàng cho Search: {file_ket_qua}")
    print("-" * 50)

if __name__ == "__main__":
    tao_vector_he_thong()
