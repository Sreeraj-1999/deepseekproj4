# vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os

class VectorStore:
    def __init__(self, db_path: str = "./vector_db"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        print("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("equipment_troubleshooting")
            print(f"Loaded existing collection with {self.collection.count()} items")
        except:
            self.collection = self.client.create_collection(
                name="equipment_troubleshooting",
                metadata={"hnsw:space": "cosine"}
            )
            print("Created new collection")
    
    # In vector_store.py, update the add_documents method:

    def add_documents(self, chunks: List[Dict]) -> Dict:
        """Add troubleshooting chunks to vector DB"""
        if not chunks:
            return {"status": "error", "message": "No chunks to add"}
        
        print(f"Adding {len(chunks)} chunks to vector DB...")
        
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [
            {
                "equipment": chunk['equipment'],
                "issue": chunk['issue'],
                "solution": chunk['solution'],
                "suspected_parts": chunk.get('suspected_parts', '')  # Add this
            }
            for chunk in chunks
        ]
        ids = [f"chunk_{i}" for i in range(self.collection.count(), self.collection.count() + len(chunks))]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {len(chunks)} chunks. Total in DB: {self.collection.count()}")
        
        return {
            "status": "success",
            "chunks_added": len(chunks),
            "total_chunks": self.collection.count()
        }
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant troubleshooting info"""
        print(f"Searching for: {query}")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        relevant_docs = []
        
        if results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                relevant_docs.append({
                    "equipment": results['metadatas'][0][i]['equipment'],
                    "issue": results['metadatas'][0][i]['issue'],
                    "solution": results['metadatas'][0][i]['solution'],
                    "text": results['documents'][0][i],
                    "score": 1 - results['distances'][0][i]  # Convert distance to similarity
                })
        
        print(f"Found {len(relevant_docs)} relevant documents")
        return relevant_docs
    
    def clear_database(self) -> Dict:
        """Clear all data from vector DB"""
        try:
            self.client.delete_collection("equipment_troubleshooting")
            self.collection = self.client.create_collection(
                name="equipment_troubleshooting",
                metadata={"hnsw:space": "cosine"}
            )
            return {"status": "success", "message": "Database cleared"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_stats(self) -> Dict:
        """Get vector DB statistics"""
        return {
            "total_chunks": self.collection.count(),
            "status": "ready"
        }