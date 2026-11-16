import os
import pickle
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# 1. Page Config & Required AI Clone Banner
st.set_page_config(page_title="Aima's Digital Clone", page_icon="🤖", layout="wide")
st.warning("⚠️ **This is an AI clone, not the real person.**")
st.title("💬 Chat with Aima's Digital Clone")

# 2. Load Precomputed Chunks & Embeddings (Cached for speed)
@st.cache_resource
def load_rag_assets():
    with open("chunks_meta.pkl", "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    metadata = data["metadata"]
    embeddings = np.load("embeddings.npy")
    
    # Normalize embeddings for fast cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / np.maximum(norms, 1e-10)
    
    # Build BM25 Index
    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Load Dense Model for query encoding
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return chunks, metadata, norm_embeddings, bm25, embed_model

chunks, metadata, norm_embeddings, bm25, embed_model = load_rag_assets()

# 3. Retrieval Functions
def retrieve_bm25(query, top_k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]

def retrieve_dense(query, top_k=3):
    q_emb = embed_model.encode([query])[0]
    q_emb = q_emb / np.maximum(np.linalg.norm(q_emb), 1e-10)
    scores = np.dot(norm_embeddings, q_emb)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]

def retrieve_hybrid_rrf(query, top_k=3, k=60):
    bm25_res = retrieve_bm25(query, top_k=len(chunks))
    dense_res = retrieve_dense(query, top_k=len(chunks))
    
    rrf_scores = {}
    for rank, (idx, _) in enumerate(bm25_res):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(dense_res):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        
    sorted_res = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(int(idx), float(score)) for idx, score in sorted_res]

# 4. LLM Generation Function
def generate_answer(query, retrieved_items):
    context_text = "\n\n---\n\n".join(
        [f"[Source: {metadata[idx]['source']} | Chunk {metadata[idx]['chunk_id']}]\n{chunks[idx]}" for idx, _ in retrieved_items]
    )
    
    prompt = f"""You are the AI clone of Aima, a 21-year-old Software Engineering student at UMT Sialkot.
Speak in the first person ("I", "my"), in a friendly, clear, and natural student tone.

Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, or private details.
4. If asked whether you are the real person, say you are an AI clone of Aima.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return "⚠️ `HF_TOKEN` environment variable not set! Pleastreamlit run app.pyse add it in Railway Variables.\n\n**Top Retrieved Info:**\n" + chunks[retrieved_items[0][0]]

    try:
        client = InferenceClient(token=hf_token)
        response = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}\n\n**Fallback from Top Chunk:** {chunks[retrieved_items[0][0]]}"

# 5. UI Controls
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_input("Ask me anything about my studies, projects, or background:", placeholder="e.g., What is your roll number? or How does Hybrid RAG work?")
    mode = st.radio("Select Retrieval Mode:", ["Hybrid (RRF)", "BM25 Only", "Dense Only"], horizontal=True)
    ask_btn = st.button("Ask Clone", type="primary")

if ask_btn and question.strip():
    with st.spinner("Searching notes & thinking..."):
        if mode == "BM25 Only":
            results = retrieve_bm25(question, top_k=3)
        elif mode == "Dense Only":
            results = retrieve_dense(question, top_k=3)
        else:
            results = retrieve_hybrid_rrf(question, top_k=3)
            
        answer = generate_answer(question, results)
        
    with col1:
        st.subheader("🤖 Answer")
        st.write(answer)
        
    with col2:
        st.subheader("📄 Retrieved Chunks & Scores")
        for idx, score in results:
            with st.expander(f"Source: {metadata[idx]['source']} (Chunk {metadata[idx]['chunk_id']}) | Score: {score:.4f}", expanded=True):
                st.write(chunks[idx])