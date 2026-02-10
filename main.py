import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Hide GPU 1, only show GPU 0
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
import logging
from typing import List, Dict, Optional, Any
import traceback
import tempfile
from pathlib import Path
import shutil
import torch
import requests
# Import all our components
from vessel_manager import VesselSpecificManager
from database import initialize_database, get_database_manager
from queue_manager import initialize_queue_manager, get_queue_manager, Priority, process_chat_immediately
from test import process_fixed_manual_query

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Marine Engineering AI System",
    description="Vessel-specific marine engineering assistance with AI",
    version="2.0.0"
)

# Global managers
vessel_manager: VesselSpecificManager = None
db_manager = None
queue_manager = None

# File upload settings
UPLOAD_FOLDER = './temp_uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.on_event("startup")
async def startup_event():
    """Initialize all components at startup"""
    global vessel_manager, db_manager, queue_manager
    
    logger.info("Starting Marine Engineering AI System...")
    
    # Create upload folder
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize vessel manager
    vessel_manager = VesselSpecificManager()
    logger.info("Vessel manager initialized")
    
    # Initialize database
    db_manager = initialize_database()
    
    # Test database connection
    if db_manager.test_connection():
        logger.info("Database connection successful")
    else:
        logger.warning("Database connection failed - alarm caching disabled")
    
    # Initialize queue manager
    queue_manager = initialize_queue_manager("http://localhost:5005")
    logger.info("Queue manager initialized")
    
    logger.info("All systems initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Marine Engineering AI System...")
    
    # Cleanup temp files
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    
    logger.info("Shutdown complete")

# ============= VESSEL MANAGEMENT =============

@app.get("/vessels/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "vessel_manager": "initialized" if vessel_manager else "not_initialized",
        "database": "connected" if db_manager and db_manager.test_connection() else "disconnected",
        "queue_manager": "initialized" if queue_manager else "not_initialized",
        "queue_status": queue_manager.get_queue_status() if queue_manager else None
    }

# ============= EXCEL TAG MANAGEMENT =============

@app.post("/vessels/{imo}/tags/upload")
async def upload_vessel_tags(imo: str, file: UploadFile = File(...)):
    """Upload tags Excel for specific vessel"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files allowed")
        
        # Save to temp location
        temp_path = os.path.join(UPLOAD_FOLDER, f"tags_{imo}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get vessel instance and upload
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.upload_tags_excel(temp_path)
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error uploading tags for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/vessels/{imo}/tags")
async def delete_vessel_tags(imo: str):
    """Delete tags for specific vessel"""
    try:
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.delete_tags_excel()
        return result
        
    except Exception as e:
        logger.error(f"Error deleting tags for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= MANUAL MANAGEMENT =============

@app.post("/vessels/{imo}/manuals/upload")
async def upload_vessel_manual(imo: str, file: UploadFile = File(...)):
    """Upload manual for specific vessel"""
    try:
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
            )
        
        # Save to temp location
        temp_path = os.path.join(UPLOAD_FOLDER, f"manual_{imo}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Add manual upload to queue (low priority)
        task_id = await queue_manager.add_task(
            task_type="manual_upload",
            endpoint="local_manual_processing",  # Special marker for local processing
            payload={"vessel_imo": imo, "file_path": temp_path},
            priority=Priority.MANUAL_UPLOAD,
            vessel_imo=imo
        )
        
        # Process locally (not GPU operation)
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.upload_manual(temp_path)
        torch.cuda.empty_cache()
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error uploading manual for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vessels/{imo}/manuals")
async def list_vessel_manuals(imo: str):
    """List all manuals for specific vessel"""
    try:
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.list_manuals()
        return result
        
    except Exception as e:
        logger.error(f"Error listing manuals for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/vessels/{imo}/manuals/{filename}")
async def delete_vessel_manual(imo: str, filename: str):
    """Delete specific manual for vessel"""
    try:
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.delete_manual(filename)
        return result
        
    except Exception as e:
        logger.error(f"Error deleting manual for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= ALARM ANALYSIS =============

@app.post("/vessels/{imo}/alarms/analyze")
async def analyze_vessel_alarms(imo: str, request_data: Dict[str, Any]):
    """Analyze alarms for specific vessel with caching"""
    try:
        alarm_list = request_data.get('alarm_name') or request_data.get('alarm') or []
        
        if not isinstance(alarm_list, list) or not alarm_list:
            raise HTTPException(status_code=400, detail="No valid alarm_name list provided")
        
        vessel = vessel_manager.get_vessel_instance(imo)
        if not vessel.tag_matcher:
         raise HTTPException(status_code=400, detail=f'No tags uploaded for vessel {imo}. Please upload tags first.')
        response_data = []
        
        for alarm_name in alarm_list:
            alarm_name = str(alarm_name).strip()
            if not alarm_name:
                continue
            
            logger.info(f"Analyzing alarm for vessel {imo}: {alarm_name}")
            
            # Check cache first
            cached_result = db_manager.check_alarm_cache(imo, alarm_name)
            
            if cached_result:
                logger.info(f"Using cached result for vessel {imo}, alarm: {alarm_name}")
                response_data.append(cached_result)
                continue
            
            # Generate new analysis
            logger.info(f"Generating new analysis for vessel {imo}, alarm: {alarm_name}")
            
            # Create system prompt for alarm analysis
            ALARM_SYSTEM_PROMPT = """Marine engineering expert. Analyze alarms precisely.

Rules:
1. Maximum 5-7 points only
2. Each point under 20 words
3. Use numbered list (1. 2. 3.)
4. Be technical and direct
5. No repetition of any phrase or sentence

For POSSIBLE REASONS:
- State ONLY what failed/went wrong
- NO action words (check, verify, test, inspect)
- Good: "Power supply failure" 
- Bad: "Check power supply"

For CORRECTIVE ACTIONS:
- State ONLY fixing steps
- Use action verbs (check, replace, verify, test)
- Good: "Replace faulty sensor"
- Bad: "Sensor malfunction"

Stop after 7 points."""

            # Generate possible reasons
            reasons_prompt = f"Alarm: {alarm_name}. List ONLY what failed or went wrong. NO actions or fixes."
            reasons_messages = [
                {'role': 'system', 'content': ALARM_SYSTEM_PROMPT},
                {'role': 'user', 'content': reasons_prompt}
            ]
            
            # Generate corrective actions
            actions_prompt = f"Alarm: {alarm_name}. List ONLY steps to fix the problem."
            actions_messages = [
                {'role': 'system', 'content': ALARM_SYSTEM_PROMPT},
                {'role': 'user', 'content': actions_prompt}
            ]
            
            # Process reasons (queued)
            def generate_llm_response_sync(messages, response_type):
                
                response = requests.post("http://localhost:5005/gpu/llm/generate", 
                                    json={"messages": messages, "response_type": response_type}, 
                                    timeout=60)
                return response.json().get('response', '')
            
            reasons_result = process_fixed_manual_query(
            processor=vessel.get_manual_processor(),
            question=reasons_prompt,
            llm_messages=reasons_messages,
            generate_llm_response_func=generate_llm_response_sync
            )
            
            # Process actions (queued)
            actions_result = process_fixed_manual_query(
            processor=vessel.get_manual_processor(),
            question=actions_prompt, 
            llm_messages=actions_messages,
            generate_llm_response_func=generate_llm_response_sync
            )
            # print(f"DEBUG - reasons_result: {reasons_result}")
            # print(f"DEBUG - actions_result: {actions_result}")
            
            # Wait for results
            reasons_answer = reasons_result['answer']
            actions_answer = actions_result['answer']
            # print(f"DEBUG - reasons_answer: '{reasons_answer}'")
            # print(f"DEBUG - actions_answer: '{actions_answer}'")
            reasons_metadata = reasons_result.get('metadata', [])
            actions_metadata = actions_result.get('metadata', [])
            
            if not reasons_result or 'error' in reasons_result:
                logger.error(f"Failed to generate reasons for alarm {alarm_name}")
                continue
                
            if not actions_result or 'error' in actions_result:
                logger.error(f"Failed to generate actions for alarm {alarm_name}")
                continue
            
            # reasons_answer = reasons_result.get('response', '')
            # actions_answer = actions_result.get('response', '')
            
            # Enhanced alarm analysis with vessel tags
            enhanced_result = vessel.analyze_alarm(alarm_name, reasons_answer, actions_answer)

            # print(f"DEBUG - enhanced_result from vessel.analyze_alarm: {enhanced_result}")
            # enhanced_result['possible_reasons'] = reasons_answer
            # enhanced_result['corrective_actions'] = actions_answer
            # print(f"DEBUG - enhanced_result after manual assignment: {enhanced_result}")
            # enhanced_result['reasons_metadata'] = reasons_metadata
            # enhanced_result['actions_metadata'] = actions_metadata
            if 'error' not in enhanced_result:
                enhanced_result['possible_reasons'] = reasons_answer
                enhanced_result['corrective_actions'] = actions_answer
                enhanced_result['reasons_metadata'] = reasons_metadata
                enhanced_result['actions_metadata'] = actions_metadata
            else:
                # Don't add anything to error response
                pass
            
            # Store in cache
            db_manager.store_alarm_analysis(
                vessel_imo=imo,
                alarm_name=alarm_name,
                possible_reasons=reasons_answer,
                corrective_actions=actions_answer,
                suspected_tags=enhanced_result.get('suspected_tags', []),
                metadata={
                'analysis_type': 'ai_generated',
                'model_version': '2.0',
                'reasons_sources': reasons_metadata,  # Add this
                'actions_sources': actions_metadata   # Add this
                }
                # metadata={
                #     'analysis_type': 'ai_generated',
                #     'model_version': '2.0'
                # }
            )
            # print(f"DEBUG - final enhanced_result before append: {enhanced_result}")
            response_data.append(enhanced_result)
            # print(f"####################DEBUG - final response_data: {response_data}")
        return {'data': response_data}
        
    except Exception as e:
        # logger.error(f"Error analyzing alarms for vessel {imo}: {e}")
        # raise HTTPException(status_code=500, detail=str(e))
        
        logger.error(f"Error analyzing alarms for vessel {imo}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= MANUAL QUERY =============

@app.post("/vessels/{imo}/manuals/query")
async def query_vessel_manuals(imo: str, request_data: Dict[str, Any]):
    """Query manuals for specific vessel"""
    try:
        question = (request_data.get('question') or request_data.get('query') or '').strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        logger.info(f"Manual query for vessel {imo}: {question}")
        
        vessel = vessel_manager.get_vessel_instance(imo)
        
        # Build messages for LLM
        SYSTEM_PROMPT = """You are a senior marine engineering assistant. Keep responses concise and under 200 words. Be technical and direct. Always end with a complete sentence and full stop."""
        
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question}
        ]
        
        # Add to medium priority queue
        task_id = await queue_manager.add_task(
            task_type="manual_query",
            endpoint="local_manual_query",  # Special marker for local processing
            payload={"vessel_imo": imo, "question": question, "messages": messages},
            priority=Priority.MANUAL_QUERY,
            vessel_imo=imo
        )
        
        # Process locally using vessel-specific manuals
        def generate_llm_response(messages, response_type):
            # This will be called by process_fixed_manual_query
            # We need to make a GPU service call
            # import requests
            try:
                response = requests.post(
                    "http://localhost:5005/gpu/llm/generate",
                    json={"messages": messages, "response_type": response_type},
                    timeout=60
                )
                response.raise_for_status()
                print(f"DEBUG - GPU service response: {result}")
                result = response.json()
                return result.get('response', '')
            except Exception as e:
                logger.error(f"GPU service call failed: {e}")
                return "I apologize, but I'm experiencing technical difficulties. Please try again later."
        
        # Process query with vessel-specific manual context
        result = process_fixed_manual_query(
            processor=vessel.get_manual_processor(),
            question=question,
            llm_messages=messages,
            generate_llm_response_func=generate_llm_response
        )
        
        # Add vessel info
        result['vessel_imo'] = imo
        
        return result
        
    except Exception as e:
        logger.error(f"Error querying manuals for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= CHAT OPERATIONS (HIGHEST PRIORITY) =============

@app.post("/vessels/{imo}/chat")
async def chat_with_vessel(imo: str, request_data: Dict[str, Any]):
    """Chat with AI using vessel-specific context (immediate processing)"""
    try:
        question = (request_data.get('chat') or request_data.get('question') or '').strip()
        # with_audio = request_data.get('with_audio', False)
        
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        logger.info(f"Chat request for vessel {imo}: {question}")
        
        # Process chat query with vessel context
        vessel = vessel_manager.get_vessel_instance(imo)
        
        SYSTEM_PROMPT = """You are a senior marine engineering assistant. Keep responses concise and under 200 words. Be technical and direct. Always end with a complete sentence and full stop."""
        
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question}
        ]
        
        # Process locally using vessel-specific manuals
        def generate_llm_response_sync(messages, response_type):
            # Make synchronous call to GPU service for immediate processing
            import requests
            try:
                response = requests.post(
                    "http://localhost:5005/gpu/llm/generate",
                    json={"messages": messages, "response_type": response_type},
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                return result.get('response', '')
            except Exception as e:
                logger.error(f"GPU service call failed: {e}")
                return "I apologize, but I'm experiencing technical difficulties. Please try again later."
        
        # Process with vessel-specific context
        result = process_fixed_manual_query(
            processor=vessel.get_manual_processor(),
            question=question,
            llm_messages=messages,
            generate_llm_response_func=generate_llm_response_sync
        )
        
        # Add audio if requested
        # if with_audio and result.get('answer'):                                  #CHANGED
        # if result.get('answer'):                            
        #     audio_task_id = await queue_manager.add_task(
        #         task_type="text_to_speech",
        #         endpoint="/gpu/tts/generate",
        #         payload={"text": result['answer']},
        #         priority=Priority.AUDIO,
        #         vessel_imo=imo
        #     )
            
        #     audio_result = await queue_manager.get_task_result_async(audio_task_id)
        #     if audio_result and not audio_result.get('error'):
        #         result['audio_blob'] = audio_result.get('audio_blob')
        if result.get('answer'):
            # Process audio immediately, bypass queue
            audio_result = await queue_manager.process_immediately(
                task_type="text_to_speech",
                endpoint="/gpu/tts/generate", 
                payload={"text": result['answer']},
                vessel_imo=imo
            )
            
            if audio_result and not audio_result.get('error'):
                result['audio_blob'] = audio_result.get('audio_blob')
        
        # Add vessel info
        result['vessel_imo'] = imo
        
        return result
        
    except Exception as e:
        logger.error(f"Error in chat for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= AUDIO OPERATIONS =============

@app.post("/vessels/{imo}/audio/transcribe")
async def transcribe_audio(imo: str, audio: UploadFile = File(...)):
    """Transcribe audio for specific vessel"""
    try:
        # Add to high priority queue
        # task_id = await queue_manager.add_task(                                        
        #     task_type="audio_transcription",
        #     endpoint="/gpu/stt/transcribe",
        #     payload={"files": {"audio": audio.file}},
        #     priority=Priority.AUDIO,
        #     vessel_imo=imo
        # )
        
        # result = await queue_manager.get_task_result_async(task_id)                ##### CHANGED
        result = await queue_manager.process_immediately(
        task_type="audio_transcription",
        endpoint="/gpu/stt/transcribe",
        payload={"files": {"audio": audio.file}},
        vessel_imo=imo
    )
        
        if result and not result.get('error'):
            result['vessel_imo'] = imo
            
        return result
        
    except Exception as e:
        logger.error(f"Error transcribing audio for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vessels/{imo}/audio/chat")
async def audio_chat(imo: str, audio: UploadFile = File(...)):
    """Process audio chat for specific vessel"""
    try:
        # First transcribe (high priority)                             ########## changed
        # transcribe_task_id = await queue_manager.add_task(
        #     task_type="audio_transcription",
        #     endpoint="/gpu/stt/transcribe", 
        #     payload={"files": {"audio": audio.file}},
        #     priority=Priority.AUDIO,
        #     vessel_imo=imo
        # )
        
        # transcribe_result = await queue_manager.get_task_result_async(transcribe_task_id)
        transcribe_result = await queue_manager.process_immediately(
        task_type="audio_transcription",
        endpoint="/gpu/stt/transcribe", 
        payload={"files": {"audio": audio.file}},
        vessel_imo=imo
         )
        
        if not transcribe_result or transcribe_result.get('error'):
            raise HTTPException(status_code=500, detail="Audio transcription failed")
        
        transcription = transcribe_result.get('transcription', '')
        
        # Then process chat (immediate)
        chat_result = await chat_with_vessel(imo, {
            "chat": transcription,
            "with_audio": True
        })
        
        # Add transcription to result
        chat_result['transcription'] = transcription
        
        return chat_result
        
    except Exception as e:
        logger.error(f"Error in audio chat for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= FACE RECOGNITION =============

# @app.post("/vessels/{imo}/face/compare")
@app.post("/image/response/")
async def compare_faces(
    # imo: str, 
    image: UploadFile = File(...), 
    profilePicture: UploadFile = File(...)
):
    """Compare faces for specific vessel"""
    try:
        # Add to high priority queue
        task_id = await queue_manager.add_task(
            task_type="face_comparison",
            endpoint="/gpu/face/compare",
            payload={"files": {"image": image.file, "profilePicture": profilePicture.file}},
            priority=Priority.FACE_RECOGNITION,
            # vessel_imo=imo
        )
        
        result = await queue_manager.get_task_result_async(task_id)
        
        if result and not result.get('error'):
            result.pop('vessel_imo',None)
            
        return result
        
    except Exception as e:
        logger.error(f"Error comparing faces : {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= FUEL ANALYSIS =============

@app.post("/vessels/{imo}/fuel/analyze")
async def analyze_fuel_consumption(imo: str, request_data: Dict[str, Any]):
    """Analyze fuel consumption for specific vessel"""
    try:
        # Add to medium priority queue
        task_id = await queue_manager.add_task(
            task_type="fuel_analysis",
            endpoint="/gpu/fuel/analyze",
            payload=request_data,
            priority=Priority.FUEL_ANALYSIS,
            vessel_imo=imo
        )
        
        result = await queue_manager.get_task_result_async(task_id, timeout=120)
        
        if result and not result.get('error'):
            result['vessel_imo'] = imo
            
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing fuel for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= SYSTEM MONITORING =============

@app.get("/system/status")
async def system_status():
    """Get overall system status"""
    try:
        db_stats = db_manager.get_database_stats() if db_manager else {"status": "not_initialized"}
        queue_status = queue_manager.get_queue_status() if queue_manager else {"status": "not_initialized"}
        
        return {
            "system": "Marine Engineering AI v2.0",
            "status": "operational",
            "database": db_stats,
            "queue": queue_status,
            "vessel_cache_size": len(vessel_manager._vessel_cache) if vessel_manager else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vessels/{imo}/history")
async def get_vessel_alarm_history(imo: str, limit: int = 50):
    """Get alarm analysis history for specific vessel"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        history = db_manager.get_vessel_alarm_history(imo, limit)
        
        return {
            "vessel_imo": imo,
            "alarm_history": history,
            "total_records": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting history for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= STARTUP =============
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI Marine Engineering AI System...")
    uvicorn.run(app, host="0.0.0.0", port=5004)