import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import argparse
import os

class GATRefiner(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8):
        super(GATRefiner, self).__init__()
        # Lớp GAT: Sử dụng Attention để tổng hợp thông tin từ hàng xóm
        self.gat1 = GATConv(in_channels, out_channels, heads=heads, concat=False)
        self.gat2 = GATConv(out_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # x: Đặc trưng từ SPECTER (768 chiều)
        # edge_index: Các liên kết trích dẫn
        x = self.gat1(x, edge_index)
        x = F.elu(x) # Hàm kích hoạt ELU cho GAT
        x = self.gat2(x, edge_index)
        return x

def build_graph(embeddings_path, metadata_path):
    print("--- 1. Đang xây dựng Đồ thị trích dẫn ---")
    
    # Load Embeddings từ SPECTER
    paper_ids = []
    embeds = []
    with open(embeddings_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            paper_ids.append(d['paper_id'])
            embeds.append(d['embedding'])
    
    id_map = {pid: i for i, pid in enumerate(paper_ids)}
    x = torch.tensor(embeds, dtype=torch.float)

    # Load Quan hệ trích dẫn
    edges = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        # Hỗ trợ cả hai định dạng (List hoặc Dict)
        paper_list = metadata if isinstance(metadata, list) else metadata.values()
        
        for item in paper_list:
            src_id = item['paper_id']
            if src_id not in id_map: continue
            
            src_idx = id_map[src_id]
            for cite in item.get('citations', []):
                dst_id = cite['paperId']
                if dst_id in id_map:
                    dst_idx = id_map[dst_id]
                    # Thêm cạnh hai chiều để thông tin truyền đi tốt hơn (hoặc một chiều tùy bài toán)
                    edges.append([src_idx, dst_idx])
                    edges.append([dst_idx, src_idx])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Đồ thị: {len(paper_ids)} Nodes, {edge_index.shape[1]} Edges.")
    
    return Data(x=x, edge_index=edge_index), paper_ids

def train_gat(data, epochs=100):
    print("--- 2. Đang huấn luyện lớp GAT (Message Passing) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATRefiner(in_channels=data.x.shape[1], out_channels=768).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    model.train()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Loss: Ép các node có liên kết gần nhau hơn trong ko gian vector
        # Đây là một dạng self-supervised learning đơn giản
        pos_loss = F.mse_loss(out[data.edge_index[0]], out[data.edge_index[1]])
        loss = pos_loss
        
        loss.backward()
        optimizer.step()
        
    print(f"Huấn luyện xong! Loss cuối cùng: {loss.item():.6f}")
    return model, data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True, help='Path to spectral embeddings .jsonl')
    parser.add_argument('--metadata', required=True, help='Path to metadata .json')
    parser.add_argument('--output', required=True, help='Path to save refined embeddings .jsonl')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # 1. Xây dựng đồ thị
    graph_data, paper_ids = build_graph(args.embeddings, args.metadata)
    
    # 2. Huấn luyện GAT
    if graph_data.edge_index.shape[1] > 0:
        model, data = train_gat(graph_data, args.epochs)
        model.eval()
        with torch.no_grad():
            refined_embeds = model(data.x, data.edge_index).cpu().numpy()
    else:
        print("Cảnh báo: Không tìm thấy cạnh trích dẫn nào trong tập dữ liệu! Giữ nguyên vector.")
        refined_embeds = graph_data.x.numpy()

    # 3. Lưu kết quả
    print(f"--- 3. Đang lưu Vector đã nâng cấp GAT vào {args.output} ---")
    with open(args.output, 'w') as f:
        for i, pid in enumerate(paper_ids):
            res = {"paper_id": pid, "embedding": refined_embeds[i].tolist()}
            f.write(json.dumps(res) + '\n')
    print("HOÀN THÀNH!")

if __name__ == "__main__":
    main()
