from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def encodeChunks(chunks):
    encoded_chunks = []
    for chunk in chunks:
        encoded_chunks.append(model.encode(chunk).tolist())
        
    return encoded_chunks