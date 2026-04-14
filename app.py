import streamlit as st
import pandas as pd
import os
import pickle
import json
import streamlit.components.v1 as components
import sys

# Đảm bảo import được các module local
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from search_module import GraphSearchEngine
from visualize_kg import visualize_kg, visualize_local_kg
from kg_builder import extract_knowledge

# --- CONFIG ---
st.set_page_config(page_title="Graph-RAG Academic Intelligence", layout="wide", page_icon="🎓")

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
    .main { background-color: #0F172A; color: #E2E8F0; font-family: 'Inter', sans-serif; }
    .stButton>button { width: 100%; border-radius: 8px; background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); color: white; border: none; padding: 12px; font-weight: 600; transition: all 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); }
    .result-card { padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 12px; background-color: #1E293B; border-left: 5px solid #3B82F6; }
    .legend-item { display: flex; align-items: center; margin-bottom: 8px; font-size: 0.9em; }
    .legend-color { width: 14px; height: 14px; border-radius: 3px; margin-right: 10px; }
    .intro-box { background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 10px; padding: 15px; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    root_dir = os.path.dirname(current_dir)
    db_file = os.path.join(root_dir, 'graph_vector.pkl')
    kg_file = os.path.join(root_dir, 'knowledge_graph.pkl')
    return GraphSearchEngine(db_file=db_file, kg_file=kg_file)

@st.cache_data
def load_all_papers():
    root_dir = os.path.dirname(current_dir)
    path = os.path.join(root_dir, "papers_clean.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def run_app():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🎓 Graph-RAG AI")
        st.markdown("""
        <div class="intro-box">
        <b>Chào bạn!</b> Đây là hệ thống thông minh giúp bạn khám phá tri thức từ hàng nghìn bài báo khoa học. Công nghệ <b>Graph-RAG</b> giúp máy tính không chỉ tìm kiếm mà còn "vẽ bản đồ" các ý tưởng.
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🎨 Chú giải Bản đồ")
        st.markdown(f"""
        <div class="legend-item"><div class="legend-color" style="background:#3B82F6"></div>Concept (Khái niệm chung)</div>
        <div class="legend-item"><div class="legend-color" style="background:#F59E0B"></div>Entity (Thực thể khoa học)</div>
        <div class="legend-item"><div class="legend-color" style="background:#10B981"></div>Chemical/Method (Phương pháp)</div>
        <div class="legend-item"><div class="legend-color" style="background:#EF4444"></div>Disease/Issue (Vấn đề)</div>
        <div class="legend-item"><div class="legend-color" style="background:#8B5CF6"></div>Genetic (Nguồn gốc)</div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📖 Hướng dẫn nhanh")
        st.markdown("""
        1. **Library Search**: Nhập từ khóa để tìm bài báo liên quan.
        2. **Document Insight**: Chọn 1 bài báo và bấm "Phân tích" để xem bản đồ ý tưởng.
        """)

    # --- MAIN CONTENT ---
    tab1, tab2 = st.tabs(["🔍 Library Search", "🧬 Document Insight"])

    with tab1:
        st.header("Search & Discovery")
        st.markdown("Hệ thống sẽ dựa trên nội dung và mạng lưới trích dẫn để tìm ra các tài liệu phù hợp nhất cho nghiên cứu của bạn.")
        
        engine = load_engine()
        col_s1, col_s2 = st.columns([1, 1.5])
        
        with col_s1:
            query = st.text_input("Gõ từ khóa hoặc câu hỏi của bạn:", placeholder="GNN applications in health...")
            if query:
                with st.spinner("Đang tìm bài báo..."):
                    results = engine.search(query, top_k=5)
                    st.success(f"Tìm thấy {len(results)} tài liệu giá trị.")
                    for i, res in enumerate(results):
                        st.markdown(f"""
                        <div class="result-card">
                            <small style="color: #60A5FA;">#{i+1} [Relevance: {res['score']:.3f}]</small><br>
                            <b>{res['title']}</b>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col_s2:
            if query:
                st.subheader("🌐 Global Citation Graph")
                root_dir = os.path.dirname(current_dir)
                kg_path = os.path.join(root_dir, "knowledge_graph.pkl")
                visualize_kg(kg_path, "temp_graph.html")
                with open("temp_graph.html", 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=650)
            else:
                st.info("Nhập từ khóa bên trái để bắt đầu khám phá.")

    with tab2:
        st.header("Deep Document Understanding")
        st.markdown("Chế độ này giúp bạn 'đọc hiểu' sâu 1 bài báo bằng cách bóc tách các thực thể và quan hệ thực tế bên trong văn bản.")
        
        papers = load_all_papers()
        if not papers:
            st.warning("⚠️ Chưa nạp được dữ liệu bài báo.")
        else:
            paper_titles = {p['title']: p for p in papers}
            selected_title = st.selectbox("Chọn hoặc tìm một bài báo để phân tích:", list(paper_titles.keys()))
            selected_p = paper_titles[selected_title]
            
            with st.expander("📄 Xem tóm tắt (Abstract)"):
                st.write(selected_p.get('abstract', 'Không có tóm tắt.'))
            
            if st.button("🚀 Bắt đầu Phân tích Thực thể"):
                with st.spinner("Đang xây dựng bản đồ khái niệm..."):
                    text = selected_p['title'] + " " + selected_p.get('abstract', '')
                    knowledge = extract_knowledge(text, top_n=25)
                    visualize_local_kg(knowledge['concepts'], knowledge['relations'], "local_graph.html")
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.subheader("📍 Key Concepts")
                        for c_info in knowledge['concepts']:
                            st.markdown(f"- `{c_info['text']}`")
                        
                        st.markdown("---")
                        st.subheader("💡 Knowledge Triplet")
                        context = " | ".join([f"{s}-{v}->{o}" for s,v,o in knowledge['relations'][:5]])
                        st.write(f"Hệ thống đã kết nối: {context} ...")
                    
                    with c2:
                        st.subheader("🌐 Semantic Entity Graph")
                        with open("local_graph.html", 'r', encoding='utf-8') as f:
                            components.html(f.read(), height=600)

if __name__ == "__main__":
    run_app()
