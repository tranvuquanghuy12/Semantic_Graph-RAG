import json
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import tqdm

def load_data(embeddings_path, metadata_path):
    print("--- 1. Đang tải cơ sở dữ liệu Vector ---")
    embeddings = []
    paper_ids = []
    with open(embeddings_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            paper_ids.append(data['paper_id'])
            embeddings.append(data['embedding'])
    
    embeddings = np.array(embeddings)
    print(f"[OK] Đã tải {len(embeddings)} bản ghi.")

    print("--- 2. Đang tải thông tin bài báo (Metadata) ---")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        raw_metadata = json.load(f)
        if isinstance(raw_metadata, list):
            metadata = {item['paper_id']: item for item in raw_metadata}
        else:
            metadata = raw_metadata
    return embeddings, paper_ids, metadata

def search(query, embeddings, paper_ids, metadata, model, tokenizer, device, top_k=5):
    # Encode truy vấn
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        result = model(**inputs)
    query_vector = result.last_hidden_state[:, 0, :].cpu().numpy()

    # Tính độ tương đồng
    scores = cosine_similarity(query_vector, embeddings)[0]
    
    # Lấy Top K kết quả
    top_indices = scores.argsort()[-top_k:][::-1]

    print("\n" + "="*70)
    print(f"KẾT QUẢ TÌM KIẾM CHO: '{query}'")
    print("="*70)
    
    for rank, idx in enumerate(top_indices, 1):
        p_id = paper_ids[idx]
        score = scores[idx]
        paper_info = metadata.get(p_id, {})
        title = paper_info.get('title', 'Không có tiêu đề')
        
        print(f"Top {rank} | Độ khớp: {score:.4f} | ID: {p_id}")
        print(f"      Tiêu đề: {title}")
        print("-" * 70)

def main():
    parser = argparse.ArgumentParser(description="Demo tìm kiếm bài báo khoa học sử dụng SPECTER")
    parser.add_argument('--embeddings', required=True, help='Đường dẫn tới file .jsonl chứa vector')
    parser.add_argument('--metadata', required=True, help='Đường dẫn tới file .json chứa metadata gốc')
    parser.add_argument('--query', type=str, help='Câu hỏi tìm kiếm (nếu không nhập sẽ vào chế độ loop)')
    parser.add_argument('--top-k', type=int, default=5, help='Số lượng kết quả trả về')
    
    args = parser.parse_args()

    embeddings, paper_ids, metadata = load_data(args.embeddings, args.metadata)

    # Khởi tạo model SPECTER
    print("--- 3. Đang khởi tạo mô hình SPECTER ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter').to(device)
    model.eval()
    print(f"[OK] Đã sẵn sàng trên thiết bị: {device}")

    if args.query:
        search(args.query, embeddings, paper_ids, metadata, model, tokenizer, device, args.top_k)
    else:
        # Chế độ nhập liệu liên tục
        while True:
            user_query = input("\nNhập nội dung bạn muốn tìm (hoặc 'q' để thoát): ")
            if user_query.lower() == 'q':
                break
            search(user_query, embeddings, paper_ids, metadata, model, tokenizer, device, args.top_k)

if __name__ == '__main__':
    main()
