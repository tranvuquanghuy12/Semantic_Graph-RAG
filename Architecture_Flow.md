# KIẾN TRÚC HỆ THỐNG GRAPH-RAG (SMART-GAT)
*Tài liệu giải thích luồng hoạt động chi tiết của dự án Tìm kiếm Ngữ nghĩa lai Đồ thị Tri thức.*

---

## 🏗️ 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG (ARCHITECTURE FLOW)

Hệ thống bạn vừa xây dựng là một mô hình **Hybrid Retrieval-Augmented Generation (Graph-RAG)** tiên tiến. Nó không chỉ tìm kiếm dựa trên chữ (Semantic Search) mà còn "nhìn" thấy các liên kết khoa học (như trích dẫn, từ khóa chung) thông qua Đồ thị để lôi ra các bài báo liên quan nhất.

Kiến trúc chia làm **3 Trụ cột chính (Pillars)** hoạt động nối tiếp nhau:
1. **Pillar 1: Data Preparation & Knowledge Graph (Chuẩn bị và Dựng đồ thị thô)**
2. **Pillar 2: Deep Learning GAT Refinement (Nâng cấp Vector bằng AI GAT)**
3. **Pillar 3: Hybrid Search Engine (Động cơ Tìm kiếm Lai)**

---

## ⚙️ 2. GIẢI THÍCH CHI TIẾT TỪNG BƯỚC ĐÃ CHẠY TRÊN TERMINAL

Dưới đây là sơ đồ giải mã những lệnh bạn vừa gõ cộc cộc trên Terminal máy chủ, chúng đã làm gì ở "hậu trường" hệ thống?

### Bước 1: Khởi tạo Đồ thị Tri thức (Knowledge Graph)
🔹 **Lệnh đã gõ:**
```bash
python kg_builder.py
```
🔍 **Hậu trường:**
- **Đầu vào:** File `papers_clean.json`.
- **Luồng xử lý:** Mô-đun này đọc toàn bộ tiêu đề, tóm tắt, trích dẫn. Nó dùng NLP nhặt ra các hạt nhân ngữ nghĩa (Concepts) và vẽ các đường nối (Edges) giữa Bài báo - Bài báo (Cites), Bài báo - Từ khóa (Has_Concept).
- **Đầu ra:** Nó đóng băng toàn bộ màng nhện tri thức này thành khối băng **`knowledge_graph.pkl`**. Đây chính là "Bản đồ vệ tinh" cho hệ thống lúc tìm kiếm.

---

### Bước 2: Ép Vector Thô (Base Embeddings)
🔹 **Lệnh đã gõ:**
```bash
python scripts/embed_papers_hf.py --data-path papers_clean.json --output raw_embeddings.jsonl
```
🔍 **Hậu trường:**
- **Đầu vào:** File `papers_clean.json`.
- **Luồng xử lý:** Gửi toàn bộ Tiêu đề + Nội dung bài báo vào não bộ AI của HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`). AI nhai chữ và nhả ra mã vạch (Vector 384 chiều).
- **Đầu ra:** File **`raw_embeddings.jsonl`**. (Bạn cứ hiểu mỗi bài báo lúc này đã biến thành 1 tọa độ 384 chiều trong không gian).

---

### Bước 3: Huấn luyện GAT (Graph Attention Network) - VŨ KHÍ BÍ MẬT 🧠
🔹 **Lệnh đã gõ:**
```bash
python scripts/gat_refinement.py --embeddings raw_embeddings.jsonl --metadata papers_clean.json --output refined_embeddings.jsonl --epochs 100
```
🔍 **Hậu trường:**
- **Đầu vào:** Vector thô (Bước 2).
- **Luồng xử lý:** Nếu chỉ dùng Vector ở Bước 2, các bài báo có nội dung giống nhau sẽ đứng gần nhau, nhưng những bài báo trích dẫn nhau (đồng môn) lại có thể đứng rất xa. **GAT (Graph Attention Network)** sẽ lấy bản đồ trích dẫn (KNN Graph) và ép các bài báo "hàng xóm" xích lại gần nhau hơn trong không gian không gian Vector.
- **Đầu ra:** File **`refined_embeddings.jsonl`** và mô hình trọng số `gat_model.pt`. (Vector lúc này đã mang trong mình "Gen Trích Dẫn").

---

### Bước 4: Tích hợp GAT Vector vào CSDL Tìm Kiếm
🔹 **Lệnh đã gõ:**
```bash
python database_vector.py
```
🔍 **Hậu trường:**
- **Luồng xử lý:** Thay vì mỗi lần tìm kiếm phải load file JSONl lặt vặt mất thời gian, script này nạp cái `refined_embeddings.jsonl` (của Bước 3) vào pandas, dập thành bảng Excel gọn gàng.
- **Đầu ra:** File **`graph_vector.csv`**. (Đây là Hệ Quản trị CSDL của chúng ta).

---

### Bước 5: Cú chốt hạ - TÌM KIẾM HYBRID (Graph-RAG)
🔹 **Lệnh đã gõ:**
```bash
python search_module.py
```
🔍 **Hậu trường xuyên thấu:**
Khi có một câu hỏi query *"application of graph models in network citation and deep learning"*, 3 Giai đoạn (Phases) chạy ngầm như sau:

* **[PHASE 1: DENSE SEARCH] - Đột kích bằng Vector:**
  Cầm câu hỏi ép thành Mã Vector. Mang mã này đi đo khoảng cách Cosine với toàn bộ cái file CSV `graph_vector.csv`.
  *Kết quả:* Tóm cổ được 1 ông Top 1 có nội dung sát nhất (Ví dụ: bài *The CALCULUS research...*).

* **[PHASE 2: GRAPH-RAG EXPANSION] - Vây bắt bằng Bản đồ:**
  Mở bản đồ vệ tinh `knowledge_graph.pkl` (tạo từ Bước 1) ra. Dùng vị trí của ông Top 1, gọi hàm đồ thị truy tìm tông ti họ hàng của ông này: 
  *Ông này hay chơi với từ khóa (Concept) nào? Ông này trích dẫn ông nào?*
  *Kết quả:* Moi thêm được 3 nhân vật có liên quan mật thiết ẩn sâu bên dưới (mà Vector Search ở Phase 1 bị mù không thấy).

* **[PHASE 3: GENERATING RAG PROMPT] - Đóng gói mớm cho LLM:**
  Gom cổ cả 4 đối tượng (1 Top 1 + 3 Hàng xóm) ghép lại thành một đoạn văn bản (Context). Tạo thành một chiếc Prompt dâng tận miệng cho ChatGPT/Llama yêu cầu nó trả lời câu hỏi gốc!

---
🏆 **KẾT LUẬN**
Bằng kiến trúc này, bạn đã giải quyết triệt để vấn đề "Khoảng cách ngữ nghĩa" và "Bỏ lỡ tri thức ngầm" đã nêu ở Phần Mở đầu Đề tài. Vector GAT giải quyết Semantic Gap, và Graph-RAG (Phase 2) kéo ra các Latent Relations!
