import json
import argparse
from tqdm.auto import tqdm
import pathlib
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='path to a json file containing paper metadata')
    parser.add_argument('--output', required=True, help='path to write the output embeddings file.')
    args = parser.parse_args()

    print("--- Đang tải mô hình SentenceTransformer siêu nhẹ ---")
    # Dùng MiniLM v2 giống hệt trong file search để đảm bảo tương thích 100% và không dính lỗi bảo mật Torch
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"--- Đang nạp dữ liệu từ {args.data_path} ---")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = {}
    print("--- Đang ép mảng Vector (Embedding) ---")
    
    # Duyệt qua từng bài báo
    for item in tqdm(data, desc="Mã hóa bài báo"):
        paper_id = item['paper_id']
        title = item.get('title', '')
        abstract = item.get('abstract', '')
        
        # Gộp Title và Abstract lại để AI hiểu ngữ cảnh trọn vẹn
        text = title + " " + abstract
        
        # Mã hóa thành mảng số tự động
        emb = model.encode(text)
        
        results[paper_id] = {
            "paper_id": paper_id, 
            "embedding": emb.tolist()
        }

    # Xuất ra file
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as fout:
        for res in results.values():
            fout.write(json.dumps(res) + '\n')
            
    print(f"--- HOÀN THÀNH: Đã lưu File vào {args.output} ---")

if __name__ == '__main__':
    main()
