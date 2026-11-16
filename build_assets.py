import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

data_dir = "data"
chunks = []
metadata = []

print("Reading and chunking documents from 'data/' folder...")
if not os.path.exists(data_dir):
    print(f"Error: '{data_dir}' folder nahi mila! Pehle data folder banayein.")
else:
    for filename in os.listdir(data_dir):
        if filename.endswith(".md") or filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                
                # Split text into chunks (~250 words)
                words = text.split()
                chunk_size = 250
                overlap = 50
                
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)
                    chunks.append(chunk_text)
                    metadata.append({"source": filename, "chunk_id": len(chunks)-1})

    print(f"Total chunks created: {len(chunks)}")

    if len(chunks) > 0:
        print("Computing embeddings...")
        embeddings = model.encode(chunks, show_progress_bar=True)

        print("Saving 'embeddings.npy' and 'chunks_meta.pkl'...")
        np.save("embeddings.npy", embeddings)
        with open("chunks_meta.pkl", "wb") as f:
            pickle.dump({"chunks": chunks, "metadata": metadata}, f)

        print("SUCCESS! 'chunks_meta.pkl' aur 'embeddings.npy' ban gaye hain.")
    else:
        print("Error: Koi text/markdown file nahi mili data folder mein!")