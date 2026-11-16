import os
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load Model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Read all files from data/ folder
data_dir = "data"
chunks = []
metadata = []

print("Reading and chunking documents...")
for filename in os.listdir(data_dir):
    if filename.endswith(".md") or filename.endswith(".txt"):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
            # Simple word-based chunking (~250 words per chunk with overlap)
            words = text.split()
            chunk_size = 250
            overlap = 50
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = "عى ".join(chunk_words) # join back
                chunk_text = " ".join(chunk_words)
                chunks.append(chunk_text)
                metadata.append({"source": filename, "chunk_id": len(chunks)-1})

print(f"Total chunks created: {len(chunks)}")

# 3. Compute embeddings
print("Computing embeddings...")
embeddings = model.encode(chunks, show_progress_bar=True)

# 4. Save embeddings and chunks metadata
np.save("embeddings.npy", embeddings)
import pickle
with open("chunks_meta.pkl", "wb") as f:
    pickle.dump({"chunks": chunks, "metadata": metadata}, f)

print("Embedding preparation completed successfully! 'embeddings.npy' and 'chunks_meta.pkl' saved.")