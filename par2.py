# from sentence_transformers import SentenceTransformer

# try:
#     model = SentenceTransformer("google/embeddinggemma-300m", trust_remote_code=True)
#     print("Success! Model loaded with sentence-transformers")
    
#     # Test the specialized methods
#     query = "test query"
#     doc = "test document"
    
#     query_emb = model.encode_query(query)
#     doc_emb = model.encode_document([doc])
    
#     print(f"Query embedding shape: {query_emb.shape}")
#     print(f"Document embedding shape: {doc_emb.shape}")
    
# except Exception as e:
#     print(f"Still failing: {e}")
import torch

torch.cuda.empty_cache()
