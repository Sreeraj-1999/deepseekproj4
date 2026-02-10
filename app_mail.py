# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv

from document_processor_mail import process_document
from vector_store_mail import VectorStore
from llm_service_mail import generate_email_reply
# from llm_service_mail import generate_email_reply, detect_email_type, detect_urgency

load_dotenv()

# Initialize
app = FastAPI(title="Email RAG Bot", version="1.0")
vector_store = VectorStore(db_path=os.getenv("VECTOR_DB_PATH", "./vector_db"))
QWEN_URL = os.getenv("QWEN_API_URL", "http://localhost:5005/gpu/llm/generate")

# ============= MODELS =============

class EmailRequest(BaseModel):
    email_body: str
    sender_email: str = None           # NEW
    sender_name: str = None            # NEW
    sender_company: str = None         # NEW
    subject: str = None                # NEW

# ============= ENDPOINT 1: Upload Document =============

@app.post("/upload-manual")
async def upload_manual(file: UploadFile = File(...)):
    """
    Upload equipment troubleshooting manual (Word or Excel)
    Processes and stores in vector database
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.docx', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Only .docx, .xlsx, .xls files supported"
            )
        
        # Save uploaded file temporarily
        temp_path = f"./temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\n📄 Processing: {file.filename}")
        
        # Process document
        chunks = process_document(temp_path)
        
        if not chunks:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail="No valid content extracted from document"
            )
        
        # Add to vector store
        result = vector_store.add_documents(chunks)
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_processed": len(chunks),
            "total_in_db": result['total_chunks'],
            "message": f"Successfully processed {file.filename}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

# ============= ENDPOINT 2: Process Email =============

# ============= ENDPOINT 2: Process Email =============

# ============= ENDPOINT 2: Process Email =============

@app.post("/process-email")
async def process_email(request: EmailRequest):
    """
    Process incoming email and generate reply using RAG
    """
    try:
        email_body = request.email_body.strip()
        
        if not email_body:
            raise HTTPException(status_code=400, detail="Email body cannot be empty")
        
        # Extract sender info
        sender_email = request.sender_email or ""
        sender_name = request.sender_name or ""
        sender_company = request.sender_company or ""
        subject = request.subject or ""
        
        print(f"\n📧 Processing email...")
        print(f"From: {sender_name} <{sender_email}> ({sender_company})")
        print(f"Subject: {subject}")
        
        # Build email context
        email_context = {
            "body": email_body,
            "sender_email": sender_email,
            "sender_name": sender_name,
            "sender_company": sender_company,
            "subject": subject
        }
        
        # Search vector DB
        relevant_docs = vector_store.search(email_body, n_results=5)
        
        # Generate reply (let Qwen decide everything)
        print(f"🤖 Generating reply...")
        draft_reply = generate_email_reply(email_context, relevant_docs, QWEN_URL)
        
        print(f"✅ Done")
        
        return {
            "status": "success",
            "sender": f"{sender_name} <{sender_email}>" if sender_name else sender_email,
            "subject": subject,
            "received_email": email_body,
            "generated_reply": draft_reply,
            "relevant_knowledge": [
                {
                    "equipment": doc['equipment'],
                    "issue": doc['issue'],
                    "score": round(doc['score'], 3)
                }
                for doc in relevant_docs[:3]
            ] if relevant_docs and relevant_docs[0].get('score', 0) > 0.5 else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= UTILITY ENDPOINTS =============

@app.get("/stats")
async def get_stats():
    """Get vector database statistics"""
    return vector_store.get_stats()

@app.delete("/clear-database")
async def clear_database():
    """Clear all data from vector database"""
    result = vector_store.clear_database()
    return result

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_db": "ready",
        "total_chunks": vector_store.collection.count()
    }

# ============= RUN =============

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Email RAG Bot API...")
    print(f"📊 Vector DB: {vector_store.collection.count()} chunks loaded")
    uvicorn.run(app, host="0.0.0.0", port=8000)