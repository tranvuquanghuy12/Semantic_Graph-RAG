from pyvis.network import Network
import pickle
import os

def visualize_kg(pkl_path, output_html):
    if not os.path.exists(pkl_path):
        print(f"Lỗi: Không tìm thấy file {pkl_path}. Vui lòng chạy kg_builder.py trước.")
        return

    print(f"--- Đang tải Knowledge Graph từ {pkl_path} ---")
    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)

    # Khởi tạo Pyvis Network VIP: Bỏ các tính năng menu gây lỗi JS, giữ lại nền xịn 
    net = Network(height="900px", width="100%", bgcolor="#0F172A", font_color="#E2E8F0", directed=True)

    print("--- Đang trích xuất và tối ưu hiệu ứng đồ thị VIP ---")
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    nodes_to_show = [n for n, d in sorted_nodes[:200]] # Lấy 200 node cho hoành tráng
    
    # Tính mức độ kết nối để scale kích cỡ Node
    degree_dict = dict(G.degree)
    
    for node in nodes_to_show:
        node_attr = G.nodes[node]
        node_type = node_attr.get('type', 'paper')
        title = node_attr.get('title', str(node))
        degree = degree_dict.get(node, 1)
        
        # Format thẻ hiển thị khi di chuột vào (Tooltip)
        hover_info = f"<b>Type:</b> {node_type.upper()}<br><b>ID:</b> {node}<br><b>Connections:</b> {degree}<br><b>Title:</b> {title}"
        
        # Thiết lập style cực ngầu VIP
        if node_type == 'paper':
            node_color = {"background": "#3B82F6", "border": "#60A5FA", "highlight": {"background": "#2563EB", "border": "#93C5FD"}}
            size = 15 + (degree * 0.8)  # Trọng số độ lớn
            shape = "dot"
        else:
            node_color = {"background": "#F59E0B", "border": "#FBBF24", "highlight": {"background": "#D97706", "border": "#FDE68A"}}
            size = 25 + (degree * 1.2)  # Concept to và nổi bật hơn
            shape = "hexagon"
            
        net.add_node(node, label=title[:35]+"..." if len(title)>35 else title, 
                     title=hover_info, color=node_color, size=size, shape=shape, 
                     shadow={"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 10})

    for u, v, data in G.edges(data=True):
        if u in nodes_to_show and v in nodes_to_show:
            relation = data.get('relation', 'unknown')
            edge_color = "#94A3B8" if relation == 'cites' else "#FBBF24"
            dash_type = False if relation == 'cites' else [5, 5]
            
            # Khi di chuột vào mũi tên hiện ra quan hệ
            edge_title = f"{u} --[{relation.upper()}]--> {v}"
            
            net.add_edge(u, v, color=edge_color, title=edge_title, dashes=dash_type, 
                         width=1.5, hoverWidth=3.0, alpha=0.6)

    # Đổi sang thuật toán vật lý Barnes Hut mặc định siêu ổn định (Tránh lỗi treo 0% của thiết lập cũ)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09, overlap=0)
    
    print(f"--- Đang tạo file HTML: {output_html} ---")
    net.save_graph(output_html)
    print(f"HOÀN THÀNH! Bạn hãy mở file {output_html} bằng trình duyệt.")

if __name__ == "__main__":
    PKL_PATH = "knowledge_graph.pkl"
    HTML_PATH = "graph_visual.html"
    visualize_kg(PKL_PATH, HTML_PATH)
