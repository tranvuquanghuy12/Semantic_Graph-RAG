import json
import networkx as nx
import pickle
import os
import re
import spacy

try:
    # Cấu hình Spacy ưu tiên dùng GPU nếu có thể
    spacy.prefer_gpu()
    # Load SciSpacy model
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    print("Cảnh báo: Chưa cài đặt mô hình en_core_sci_sm.")
    nlp = None

def extract_knowledge(text, top_n=25):
    """
    Trích xuất Entity (Concepts) và Relation bản dày (Dense Version).
    Bổ sung: Co-occurrence links và Noun chunks extraction.
    """
    if not nlp:
        return {"concepts": [], "relations": []}
        
    doc = nlp(text)
    
    # 1. Trích xuất Thực thể (NER + Noun Chunks)
    entities = []
    # Ưu tiên lấy noun chunks để có cụm từ kỹ thuật đầy đủ
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 3:
            entities.append((chunk.text.lower(), "CONCEPT"))
    
    # Bổ sung NER entities
    for ent in doc.ents:
        if len(ent.text) > 3:
            entities.append((ent.text.lower(), ent.label_))

    
    # Đếm tần suất và lưu Label
    freq = {}
    label_map = {}
    for e, label in entities:
        e = re.sub(r'[^a-zA-Z\s]', '', e).strip()
        if len(e) > 3:
            freq[e] = freq.get(e, 0) + 1
            if e not in label_map:
                label_map[e] = label
            
    sorted_concepts = [item[0] for item in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    
    # Đóng gói concepts kèm label
    concept_details = [{"text": c, "label": label_map.get(c, "ENTITY")} for c in sorted_concepts]
    
    relations = []

    # 2. Quan hệ Triplet (S-V-O) - Quan hệ có nhãn
    for token in doc:
        if token.pos_ == "VERB":
            subjs = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            objs = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "attr")]
            if subjs and objs:
                s_text = subjs[0].text.lower()
                o_text = objs[0].text.lower()
                v_text = token.lemma_.lower()
                
                s_match = next((c for c in sorted_concepts if s_text in c or c in s_text), None)
                o_match = next((c for c in sorted_concepts if o_text in c or c in o_text), None)
                
                if s_match and o_match and s_match != o_match:
                    relations.append((s_match, v_text, o_match))

    # 3. Quan hệ Đồng xuất hiện (Co-occurrence) - Làm dày đồ thị
    for sent in doc.sents:
        sent_concepts = [c for c in sorted_concepts if c in sent.text.lower()]
        # Nối tất cả các cặp concept trong cùng 1 câu
        for i in range(len(sent_concepts)):
            for j in range(i + 1, len(sent_concepts)):
                relations.append((sent_concepts[i], "related_to", sent_concepts[j]))
                    
    # Lọc unique relations
    relations = list(set(relations))
    
    return {"concepts": concept_details, "relations": relations}


def build_knowledge_graph(json_path, output_path):
    print(f"--- Đang tải dữ liệu từ {json_path} ---")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.MultiDiGraph()

    for paper in data:
        pid = paper['paper_id']
        title = paper.get('title', 'Unknown Title')
        abstract = paper.get('abstract', '')
        
        G.add_node(pid, title=title, type='paper', color='#3B82F6')
        
        knowledge = extract_knowledge(title + ". " + abstract, top_n=5) # Giữ top_n thấp cho global graph để tránh lag
        for concept in knowledge["concepts"]:
            G.add_node(concept, title=concept, type='concept', color='#F59E0B')
            G.add_edge(pid, concept, relation='has_concept')
            
        for (subj, verb, obj) in knowledge["relations"]:
            G.add_edge(subj, obj, relation=verb)
        
        citations = paper.get('citations', [])
        for cite in citations:
            target_id = cite.get('paperId')
            if target_id: G.add_edge(pid, target_id, relation='cites')

    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"DONE: {output_path}")

if __name__ == "__main__":
    build_knowledge_graph("papers_clean.json", "knowledge_graph.pkl")
