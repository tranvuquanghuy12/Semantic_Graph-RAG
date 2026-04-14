from pyvis.network import Network
import pickle
import os

def visualize_kg(pkl_path, output_html):
    if not os.path.exists(pkl_path):
        return

    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)

    net = Network(height="700px", width="100%", bgcolor="#0F172A", font_color="#E2E8F0", directed=True)

    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    nodes_to_show = [n for n, d in sorted_nodes[:150]]
    degree_dict = dict(G.degree)
    
    for node in nodes_to_show:
        node_attr = G.nodes[node]
        node_type = node_attr.get('type', 'paper')
        title = node_attr.get('title', str(node))
        degree = degree_dict.get(node, 1)
        
        if node_type == 'paper':
            node_color = {"background": "#3B82F6", "border": "#60A5FA"}
            size = 15 + (degree * 0.5)
            shape = "dot"
        else:
            node_color = {"background": "#F59E0B", "border": "#FBBF24"}
            size = 20 + (degree * 0.8)
            shape = "hexagon"
            
        net.add_node(node, label=title[:30], title=title, color=node_color, size=size, shape=shape)

    for u, v, data in G.edges(data=True):
        if u in nodes_to_show and v in nodes_to_show:
            relation = data.get('relation', 'unknown')
            edge_color = "#94A3B8" if relation == 'cites' else "#FBBF24"
            net.add_edge(u, v, color=edge_color, title=relation)

    net.barnes_hut(gravity=-5000)
    net.save_graph(output_html)

def visualize_local_kg(concepts, relations, output_html):
    """
    Trực quan hóa đồ thị thực thể (Entities) và quan hệ (Relations) bản dày.
    Bổ sung: Phân loại màu sắc theo Label.
    """
    net = Network(height="650px", width="100%", bgcolor="#0F172A", font_color="#E2E8F0", directed=True)
    
    # Bảng màu Premium theo Label
    color_palette = {
        "CONCEPT": "#3B82F6", # Blue
        "ENTITY": "#F59E0B",  # Amber
        "CHEMICAL": "#10B981", # Emerald
        "DISEASE": "#EF4444",  # Rose
        "GENE": "#8B5CF6",     # Violet
        "UNKNOWN": "#94A3B8"   # Slate
    }
    
    # 1. Thêm Nodes
    for c_info in concepts:
        name = c_info['text']
        label = c_info['label']
        color = color_palette.get(label, color_palette["ENTITY"])
        
        net.add_node(name, label=name, title=f"Label: {label}", color=color, size=25, shape="hexagon",
                     shadow={"enabled": True, "size": 8})
    
    # 2. Thêm Edges

    for (s, v, o) in relations:
        is_cooccurrence = (v == "related_to")
        color = "#475569" if is_cooccurrence else "#FACC15"
        dashes = True if is_cooccurrence else False
        width = 1 if is_cooccurrence else 2
        label = "" if is_cooccurrence else v # Chỉ hiện nhãn cho quan hệ động từ chính để tránh rối
        
        net.add_edge(s, o, label=label, title=f"{s} --{v}--> {o}", color=color,
                     dashes=dashes, width=width, arrows="to" if not is_cooccurrence else "")
                     
    # Tối ưu Physics lực hút mạnh để nhìn 'đặc' hơn
    net.barnes_hut(gravity=-4000, central_gravity=0.4, spring_length=120, spring_strength=0.06)
    net.save_graph(output_html)

if __name__ == "__main__":
    visualize_kg("knowledge_graph.pkl", "graph_visual.html")
