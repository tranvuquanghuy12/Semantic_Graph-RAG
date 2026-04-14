import pandas as pd
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

def tao_vector_he_thong():
    print("=== [OPTIMIZED] HỆ THỐNG TẠO DATABASE BINARY SIÊU TỐC ===")
    
    # Ở bản này, chúng ta sẽ lưu ra file .pkl để nạp Dashboard nhanh hơn
    file_raw = 'papers_clean.json'
    file_refined = 'refined_embeddings.jsonl' 
    file_ket_qua = 'graph_vector.pkl' 
    
    if not os.path.exists(file_raw):
        print(f"[ERROR] Không tìm thấy dữ liệu thô '{file_raw}'.")
        return

    # 1. ĐỌC DỮ LIỆU GỐC
    with open(file_raw, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # 2. LUỒNG 1: TEXT EMBEDDING
    print("\n[STEP 1] Tạo Base Text Vectors (NanoNet/MiniLM)...")
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    text_data = df['title'] + " " + df['abstract'].fillna('')
    base_vectors = model.encode(text_data.tolist(), show_progress_bar=True)
    df['base_text_vector'] = [v for v in base_vectors]

    # 3. LUỒNG 2: GAT REFINED EMBEDDING
    print("\n[STEP 2] Tích hợp GAT-Refined Vectors...")
    if os.path.exists(file_refined):
        refined_data = {}
        with open(file_refined, 'r') as f:
            for line in f:
                d = json.loads(line)
                refined_data[d['paper_id']] = np.array(d['embedding'], dtype=np.float32)
        
        df['graph_vector'] = df['paper_id'].map(refined_data)
        # Fill missing with zero arrays
        df['graph_vector'] = df['graph_vector'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(768, dtype=np.float32))
        print("[OK] Đã tích hợp thành công GAT Vectors.")
    else:
        df['graph_vector'] = [np.zeros(768, dtype=np.float32) for _ in range(len(df))]

    # 4. LƯU DATABASE DƯỚI DẠNG BINARY (PICKLE)
    print("\n[STEP 3] Xuất Database Binary (Pickle)...")
    cols = ['paper_id', 'title', 'base_text_vector', 'graph_vector']
    df[cols].to_pickle(file_ket_qua)

    print("-" * 50)
    print(f"HOÀN THÀNH TỐI ƯU! File sẵn sàng cho Search: {file_ket_qua}")
    print("-" * 50)

if __name__ == "__main__":
    tao_vector_he_thong()
