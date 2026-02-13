import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Hide GPU 1, only show GPU 0
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse,StreamingResponse
import httpx
import logging
from typing import List, Dict, Optional, Any
import traceback
import os
import tempfile
from pathlib import Path
import shutil
# import datetime
from datetime import datetime
import torch
import requests
# Import all our components
from vessel_manager import VesselSpecificManager
from database import initialize_database, get_database_manager
from queue_manager import initialize_queue_manager, get_queue_manager, Priority, process_chat_immediately
# from test import process_fixed_manual_query
from test3 import process_fixed_manual_query
from faultsense import load_config_with_overrides, process_smart_maintenance_results, run_pipeline
import pickle
import pandas as pd
import json
from torch import nn
import numpy as np

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
# MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_ship_model(ship_id):
    """Load model and scalers for a specific ship"""
    base_path = r'C:\Users\User\Desktop\siemens\clemens'
    model_dir = os.path.join(base_path, "model", ship_id)
    
    try:
        # Load hyperparameters
        params_path = os.path.join(model_dir, f'FULL{ship_id}_model_params.json')
        with open(params_path, 'r') as f:
            model_params = json.load(f)
        
        # Define the neural network model with dynamic input size
        class NeuralNetwork(nn.Module):
            def __init__(self, input_size, n_layers, n_neurons, dropout_rate, l2_regularization):
                super(NeuralNetwork, self).__init__()
                layers = []
                layers.append(nn.Linear(input_size, n_neurons))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

                for _ in range(n_layers - 1):
                    layers.append(nn.Linear(n_neurons, n_neurons))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))

                layers.append(nn.Linear(n_neurons, 1))
                self.model = nn.Sequential(*layers)
                self.l2_regularization = l2_regularization

            def forward(self, x):
                return self.model(x)
        
        # Load model with dynamic parameters
        model = NeuralNetwork(
            input_size=model_params['pca_n_components'],
            n_layers=model_params['n_layers'],
            n_neurons=model_params['n_neurons'], 
            dropout_rate=model_params['dropout_rate'],
            l2_regularization=model_params['l2_regularization']
        )
        
        model_path = os.path.join(model_dir, f'FULL{ship_id}_pytorch_model.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Load scalers
        scaler_X_path = os.path.join(model_dir, f'FULL{ship_id}_scaler_X.pkl')
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        
        scaler_y_path = os.path.join(model_dir, f'FULL{ship_id}_scaler_y.pkl')
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
        
        pca_X_path = os.path.join(model_dir, f'FULL{ship_id}_pca_X.pkl')
        with open(pca_X_path, 'rb') as f:
            pca_X = pickle.load(f)
        
        return model, scaler_X, scaler_y, pca_X, device
        
    except Exception as e:
        raise FileNotFoundError(f"Model files not found for ship {ship_id}: {str(e)}")

def load_ae_model(ship_id):
    """Load AE model and scalers for a specific ship"""
    base_path = r'C:\Users\User\Desktop\siemens\clemens'
    model_dir = os.path.join(base_path, "AE_model", ship_id)
    
    try:
        # Load hyperparameters
        params_path = os.path.join(model_dir, f'FULL{ship_id}_model_params.json')
        with open(params_path, 'r') as f:
            model_params = json.load(f)
        
        # Define the neural network model with dynamic input size
        class NeuralNetwork(nn.Module):
            def __init__(self, input_size, n_layers, n_neurons, dropout_rate, l2_regularization):
                super(NeuralNetwork, self).__init__()
                layers = []
                layers.append(nn.Linear(input_size, n_neurons))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

                for _ in range(n_layers - 1):
                    layers.append(nn.Linear(n_neurons, n_neurons))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))

                layers.append(nn.Linear(n_neurons, 1))
                self.model = nn.Sequential(*layers)
                self.l2_regularization = l2_regularization

            def forward(self, x):
                return self.model(x)
        
        # Load model with dynamic parameters
        model = NeuralNetwork(
            input_size=model_params['pca_n_components'],
            n_layers=model_params['n_layers'],
            n_neurons=model_params['n_neurons'], 
            dropout_rate=model_params['dropout_rate'],
            l2_regularization=model_params['l2_regularization']
        )
        
        model_path = os.path.join(model_dir, f'FULL{ship_id}_pytorch_model.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Load scalers
        scaler_X_path = os.path.join(model_dir, f'FULL{ship_id}_scaler_X.pkl')
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        
        scaler_y_path = os.path.join(model_dir, f'FULL{ship_id}_scaler_y.pkl')
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
        
        pca_X_path = os.path.join(model_dir, f'FULL{ship_id}_pca_X.pkl')
        with open(pca_X_path, 'rb') as f:
            pca_X = pickle.load(f)
        
        return model, scaler_X, scaler_y, pca_X, device
        
    except Exception as e:
        raise FileNotFoundError(f"Model files not found for ship {ship_id}: {str(e)}")

def relative_wind_direction(wind_direction, ship_heading):
    relative_direction = wind_direction - ship_heading
    return np.mod(relative_direction + 360, 360)

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


@app.post("/vessels/{imo}/manuals/upload/")
async def upload_vessel_manual(imo: str, file: UploadFile = File(...)):
    """Upload manual for specific vessel"""
    print("IMOM", imo)
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
        # task_id = await queue_manager.add_task(
        #     task_type="manual_upload",
        #     endpoint="local_manual_processing",  # Special marker for local processing
        #     payload={"vessel_imo": imo, "file_path": temp_path},
        #     priority=Priority.MANUAL_UPLOAD,
        #     vessel_imo=imo
        # )
        # Queue task is just for logging, don't let it block upload
        try:
            task_id = await queue_manager.add_task(
                task_type="manual_upload",
                endpoint="local_manual_processing",
                payload={"vessel_imo": imo, "file_path": temp_path},
                priority=Priority.MANUAL_UPLOAD,
                vessel_imo=imo
            )
        except Exception as e:
            logger.warning(f"Queue task failed (non-blocking): {e}")

        logger.info(f"Starting manual upload for vessel {imo}")
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.upload_manual(temp_path)
        logger.info(f"Upload result: {result}")    
        
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
            
            
            # Enhanced alarm analysis with vessel tags
            enhanced_result = vessel.analyze_alarm(alarm_name, reasons_answer, actions_answer)

            
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
    


@app.post("/chat/response/")
async def chat_general(request_data: Dict[str, Any]):
    """General chat endpoint - NO RAG, NO IMO"""
    try:
        question = (request_data.get('chat') or request_data.get('question') or '').strip()
        
        if not question:
            return JSONResponse(content={'error': 'No question provided.'}, status_code=400)
        
        logger.info(f"General chat request, Q: {question}")
        
        messages = [
            {'role': 'user', 'content': f"You are a friendly and knowledgeable marine engineering assistant. Be helpful, conversational, and concise. Plain text only, no HTML. Never include special tokens or end-of-sentence markers in your response.\n\n{question}"}
        ]
        
        response = requests.post(
            "http://localhost:5005/gpu/llm/generate",
            json={"messages": messages, "response_type": "general_chat"},
            timeout=60
        )
        answer = response.json().get('response', '')
        
        data_object = {'answer': answer}
        
        # Generate audio
        if answer:
            audio_result = await queue_manager.process_immediately(
                task_type="text_to_speech",
                endpoint="/gpu/tts/generate",
                payload={"text": answer},
                vessel_imo=None
            )
            audio_blob = audio_result.get('audio_blob', '') if audio_result and not audio_result.get('error') else ''
            if audio_blob:
                data_object['blob'] = audio_blob
        
        return JSONResponse(content={'data': data_object})
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return JSONResponse(content={'error': str(e)}, status_code=500)
    


  

# @app.post("/chat/stream/")
# async def chat_stream(request_data: Dict[str, Any]):
#     """Streaming chat endpoint - SSE"""
#     question = (request_data.get('chat') or request_data.get('question') or '').strip()
#     vessel_imo = (request_data.get('imo') or '').strip()
    
#     if not question:
#         return JSONResponse(content={'error': 'No question provided.'}, status_code=400)
    
#     logger.info(f"Stream request - IMO: {vessel_imo if vessel_imo else 'GENERAL'}, Q: {question}")
    
#     async def event_generator():
#         full_answer = ""
#         try:
#             # If vessel-specific, get RAG context first
#             context = ""
#             metadata = []
#             if vessel_imo:
#                 vessel = vessel_manager.get_vessel_instance(vessel_imo)
#                 query_result = vessel.get_manual_processor().query_manuals(question, n_results=10)
#                 if query_result.get('context'):
#                     context = query_result['context']
#                     metadata = query_result.get('metadata', [])
            
            
#             if context:
#                 user_content = f"""You are a senior marine engineering assistant. Answer strictly from the provided context. Strict Rules:
#                 1. Do not add information not present in the context
#                 2. When multiple chunks are from the same document, treat them as continuous text and combine them in order
#                 3. Include ALL specific values, measurements, torque specs, and step-by-step procedures exactly as written in the context
#                 4. For procedures, list every step with full details - do not summarize or skip steps
#                 5. Plain text only, no HTML
#                 CONTEXT FROM MANUALS:
#                 {context}

#                 QUESTION: {question}"""

#             else:
#                 user_content = f"""You are a friendly and knowledgeable marine engineering assistant.  Answer in English only. Plain text only, no HTML tags or markdown. Never include special tokens or end-of-sentence markers in your response. Be concise and helpful.

#             {question}"""
            
#             # No system role — DeepSeek official recommendation
#             messages = [
#                 {'role': 'user', 'content': user_content}
#             ]
            
#             # Stream from gpu_service
#             async with httpx.AsyncClient(timeout=120.0) as client:
#                 async with client.stream(
#                     'POST',
#                     'http://localhost:5005/gpu/llm/stream',
#                     json={'messages': messages}
#                 ) as response:
#                     async for line in response.aiter_lines():
#                         if line.startswith('data:'):
#                             yield f"{line}\n\n"
                            
#                             try:
#                                 data = json.loads(line[5:].strip())
#                                 if data.get('type') == 'answer':
#                                     full_answer += data.get('content', '')
#                             except:
#                                 pass        
                    
#                     # CLEAN THE FINAL ANSWER HERE - BEFORE SENDING TO TTS
#                     if full_answer.strip():
#                         # Remove all unwanted tokens/tags
#                         full_answer = full_answer.replace('<｜end▁of▁sentence｜>', '')
#                         full_answer = full_answer.replace('<|end▁of▁sentence|>', '')
#                         full_answer = full_answer.replace('<|im_end|>', '')
#                         full_answer = full_answer.replace('<｜im_end｜>', '')
#                         full_answer = full_answer.replace('<br>', '')
#                         full_answer = full_answer.replace('</br>', '')
#                         full_answer = full_answer.replace('<br/>', '')
#                         full_answer = full_answer.strip()
#                         audio_result = await queue_manager.process_immediately(
#                             task_type="text_to_speech",
#                             endpoint="/gpu/tts/generate",
#                             payload={"text": full_answer.strip()},
#                             vessel_imo=vessel_imo if vessel_imo else None
#                         )
#                         audio_blob = audio_result.get('audio_blob', '') if audio_result and not audio_result.get('error') else ''
                        
#                         if audio_blob:
#                             yield f"data: {json.dumps({'type': 'blob', 'content': audio_blob})}\n\n"
                    
#                     if metadata:
#                         yield f"data: {json.dumps({'type': 'metadata', 'content': metadata})}\n\n"
                    
#                     yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
#         except Exception as e:
#             logger.error(f"Stream error: {e}")
#             yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
#     return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/chat/stream/")
async def chat_stream(request_data: Dict[str, Any]):
    """Streaming chat endpoint - SSE"""
    question = (request_data.get('chat') or request_data.get('question') or '').strip()
    vessel_imo = (request_data.get('imo') or '').strip()
    
    if not question:
        return JSONResponse(content={'error': 'No question provided.'}, status_code=400)
    
    logger.info(f"Stream request - IMO: {vessel_imo if vessel_imo else 'GENERAL'}, Q: {question}")
    
    async def event_generator():
        full_answer = ""
        try:
            # If vessel-specific, get RAG context first
            context = ""
            metadata = []
            if vessel_imo:
                vessel = vessel_manager.get_vessel_instance(vessel_imo)
                query_result = vessel.get_manual_processor().query_manuals(question, n_results=10)
                if query_result.get('context'):
                    context = query_result['context']
                    metadata = query_result.get('metadata', [])
            
            # Build user content
            if context:
                user_content = f"""You are a senior marine engineering assistant. Answer strictly from the provided context. Strict Rules:
1. Do not add information not present in the context
2. When multiple chunks are from the same document, treat them as continuous text and combine them in order
3. Include ALL specific values, measurements, torque specs, and step-by-step procedures exactly as written in the context
4. For procedures, list every step with full details - do not summarize or skip steps
5. Plain text only, no HTML
CONTEXT FROM MANUALS:
{context}

QUESTION: {question}"""
            else:
                user_content = f"""You are a friendly and knowledgeable marine engineering assistant. Answer in English only. Plain text only, no HTML tags or markdown. Never include special tokens or end-of-sentence markers in your response. Be concise and helpful.

{question}"""
            
            # No system role — DeepSeek official recommendation
            messages = [
                {'role': 'user', 'content': user_content}
            ]
            
            # Stream from gpu_service
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    'POST',
                    'http://localhost:5005/gpu/llm/stream',
                    json={'messages': messages}
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith('data:'):
                            try:
                                data = json.loads(line[5:].strip())
                                
                                # Clean answer chunks before sending to frontend
                                if data.get('type') == 'answer':
                                    chunk = data.get('content', '')
                                    
                                    # Clean all unwanted tokens from chunk (NO STRIP!)
                                    chunk = chunk.replace('<｜end▁of▁sentence｜>', '')
                                    chunk = chunk.replace('<|end▁of▁sentence|>', '')
                                    chunk = chunk.replace('<|im_end|>', '')
                                    chunk = chunk.replace('<｜im_end｜>', '')
                                    chunk = chunk.replace('<br>', '').replace('</br>', '').replace('<br/>', '')
                                    
                                    # Only send non-empty chunks (keep spaces!)
                                    if chunk:
                                        # Update data with cleaned chunk
                                        data['content'] = chunk
                                        
                                        # Collect for TTS
                                        full_answer += chunk
                                        
                                        # Send cleaned chunk to frontend
                                        yield f"data: {json.dumps(data)}\n\n"
                                else:
                                    # For 'thinking', 'done', 'error' - send as-is
                                    yield f"{line}\n\n"
                            except Exception as parse_error:
                                logger.warning(f"Failed to parse line: {parse_error}")
                                yield f"{line}\n\n"
                    
                    logger.info(f"STREAM COMPLETE - Full answer length: {len(full_answer)}")
                    logger.info(f"FULL ANSWER PREVIEW: {full_answer[:150]}...")
                    
                    # Generate audio from cleaned answer
                    if full_answer.strip():
                        # Final cleaning pass
                        full_answer = full_answer.replace('<｜end▁of▁sentence｜>', '')
                        full_answer = full_answer.replace('<|end▁of▁sentence|>', '')
                        full_answer = full_answer.replace('<|im_end|>', '')
                        full_answer = full_answer.replace('<｜im_end｜>', '')
                        full_answer = full_answer.replace('<br>', '').replace('</br>', '').replace('<br/>', '')
                        full_answer = full_answer.strip()  # Strip ONLY the final complete answer
                        
                        logger.info(f"GENERATING AUDIO FOR TEXT (length={len(full_answer)})")
                        
                        try:
                            audio_result = await queue_manager.process_immediately(
                                task_type="text_to_speech",
                                endpoint="/gpu/tts/generate",
                                payload={"text": full_answer},
                                vessel_imo=vessel_imo if vessel_imo else None
                            )
                            
                            logger.info(f"AUDIO RESULT RECEIVED: {list(audio_result.keys()) if audio_result else 'None'}")
                            
                            if audio_result and audio_result.get('audio_blob'):
                                audio_blob = audio_result['audio_blob']
                                blob_length = len(audio_blob)
                                
                                logger.info(f"AUDIO BLOB EXTRACTED - Length: {blob_length}")
                                
                                # Send blob IMMEDIATELY
                                blob_data = json.dumps({'type': 'blob', 'content': audio_blob})
                                logger.info(f"BLOB JSON SIZE: {len(blob_data)} bytes")
                                
                                yield f"data: {blob_data}\n\n"
                                logger.info("✓ BLOB SENT TO FRONTEND")
                            else:
                                logger.warning(f"NO AUDIO BLOB - Result: {audio_result}")
                        except Exception as audio_error:
                            logger.error(f"AUDIO GENERATION FAILED: {audio_error}")
                            logger.error(traceback.format_exc())
                    else:
                        logger.warning("FULL ANSWER IS EMPTY - NO AUDIO GENERATED")
                    
                    # Send metadata if available
                    if metadata:
                        logger.info(f"SENDING METADATA - {len(metadata)} items")
                        yield f"data: {json.dumps({'type': 'metadata', 'content': metadata})}\n\n"
                    
                    # Send done signal
                    logger.info("SENDING DONE SIGNAL")
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
        except Exception as e:
            logger.error(f"STREAM ERROR: {e}")
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/audio/transcribe/")
async def simple_transcription(audio: UploadFile = File(...)):
    """Audio transcription - old endpoint format"""
    try:
        result = await queue_manager.process_immediately(
            task_type="audio_transcription",
            endpoint="/gpu/stt/transcribe",
            payload={"files": {"audio": audio.file}},
            vessel_imo=None
        )
        
        if result and not result.get('error'):
            return JSONResponse(content={
                "success": True,
                "transcription": result.get('transcription', '')
            })
        else:
            return JSONResponse(
                content={
                    "success": False,
                    "error": result.get('error', 'Transcription failed')
                },
                status_code=500
            )
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@app.post("/vessels/chat")  # Remove {imo} from path
async def chat_general(request_data: Dict[str, Any]):
    """Chat - checks for IMO in payload, falls back to general LLM"""
    try:
        question = (request_data.get('chat') or request_data.get('question') or '').strip()
        imo = request_data.get('imo') or request_data.get('IMO')  # Get IMO from body
        
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        logger.info(f"Chat request - IMO: {imo if imo else 'GENERAL'}, Q: {question}")
        
        SYSTEM_PROMPT = """You are a senior marine engineering assistant. Keep responses concise and under 200 words. Be technical and direct. Always end with a complete sentence and full stop."""
        
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question}
        ]
        
        # If IMO provided, use vessel context
        if imo:
            vessel = vessel_manager.get_vessel_instance(str(imo))
            
            result = process_fixed_manual_query(
                processor=vessel.get_manual_processor(),
                question=question,
                llm_messages=messages,
                generate_llm_response_func=lambda msgs, rt: requests.post(
                    "http://localhost:5005/gpu/llm/generate",
                    json={"messages": msgs, "response_type": rt},
                    timeout=60
                ).json().get('response', '')
            )
            result['vessel_imo'] = imo
        else:
            # General chat - no RAG, direct LLM
            response = requests.post(
                "http://localhost:5005/gpu/llm/generate",
                json={"messages": messages, "response_type": "general_chat"},
                timeout=60
            )
            result = {
                'question': question,
                'answer': response.json().get('response', ''),
                'source': 'llm_knowledge',
                'metadata': []
            }
        
        # Add audio
        if result.get('answer'):
            audio_result = await queue_manager.process_immediately(
                task_type="text_to_speech",
                endpoint="/gpu/tts/generate",
                payload={"text": result['answer']},
                vessel_imo=imo
            )
            if audio_result and not audio_result.get('error'):
                result['audio_blob'] = audio_result.get('audio_blob')
        
        return result
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
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
        
    except Exception as e:
        logger.error(f"Error comparing faces : {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= FUEL ANALYSIS =============
@app.post("/fuel/analysis")
async def fuel_analysis(request_data: Dict[str, Any]):
    """Analyze fuel consumption - handles both ME and AE"""
    try:
        print(request_data)
        data = request_data
        
        # Handle different payload structures
        if 'fuelmasterPayload' in data:
            data = data['fuelmasterPayload']
        elif 'roundedPayloadAE' in data:
            data = data['roundedPayloadAE']
        else:
            raise HTTPException(status_code=400, detail='Invalid payload structure')
        
        # Extract ship IMO
        imo_number = data.get('IMO')
        
        if not imo_number:
            raise HTTPException(status_code=400, detail='IMO is required')
        
        ship_id = f"IMO{imo_number}" 
        
        # Try to load both models
        try:
            me_model, me_scaler_X, me_scaler_y, me_pca_X, device = load_ship_model(ship_id)
            me_available = True
        except FileNotFoundError:
            me_available = False

        try:
            ae_model, ae_scaler_X, ae_scaler_y, ae_pca_X, device = load_ae_model(ship_id)
            ae_available = True
        except FileNotFoundError:
            ae_available = False

        if not me_available and not ae_available:
            raise HTTPException(status_code=404, detail=f'No models found for {ship_id}')
        
        # Extract ME input values
        sog = data.get('V_SOG_act_kn@AVG')
        stw = data.get('V_STW_act_kn@AVG')
        rpm = data.get('SA_SPD_act_rpm@AVG')
        torque = data.get('SA_TQU_act_kNm@AVG')
        power = data.get('SA_POW_act_kW@AVG')
        actual_fuel = data.get('ME_FMS_act_kgPh@AVG')
        wind_direction = data.get('WEA_WDT_act_deg@AVG')
        ship_heading = data.get('V_HDG_act_deg@AVG')
        wind_speed = data.get('WEA_WST_act_kn@AVG')
        
        # Extract AE input values
        ae1_power = data.get('AE1_POW_act_kW@AVG')
        ae2_power = data.get('AE2_POW_act_kW@AVG')
        ae3_power = data.get('AE3_POW_act_kW@AVG')
        ae4_power = data.get('AE4_POW_act_kW@AVG')
        actual_ae_fuel = data.get('AE_HFO_FMS_act_kgPh@AVG')
        actual_ae_mdo_fuel = data.get('AE_MDO_FMS_act_kgPh@AVG')
        actual_ae_total_fuel = (actual_ae_fuel or 0) + (actual_ae_mdo_fuel or 0)
        
        # ME Prediction
        predicted_me = None
        alert_me = None
        
        required_fields = [sog, stw, rpm, torque, power, wind_direction, ship_heading, wind_speed]
        if me_available and all(x is not None for x in required_fields):
            # Calculate effective wind speed
            rel_wind_dir = relative_wind_direction(wind_direction, ship_heading)
            rel_wind_cos = np.cos(np.radians(rel_wind_dir))
            effective_wind_speed = wind_speed * rel_wind_cos
            
            # Prepare input data
            input_data = pd.DataFrame({
                'V_SOG_act_kn@AVG': [sog],
                'V_STW_act_kn@AVG': [stw], 
                'SA_SPD_act_rpm@AVG': [rpm],
                'SA_TQU_act_kNm@AVG': [torque],
                'Effective_Wind_Speed': [effective_wind_speed],
                'SA_POW_act_kW@AVG': [power]
            })
            
            # Preprocess
            X_scaled = me_scaler_X.transform(input_data)
            X_pca = me_pca_X.transform(X_scaled)
            X_tensor = torch.tensor(X_pca, dtype=torch.float32).to(device)
            
            # Predict
            with torch.no_grad():
                y_pred_scaled = me_model(X_tensor).cpu().numpy().flatten()
                
            predicted_fuel = me_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
            predicted_me = round(float(predicted_fuel), 2)
            
            if predicted_me < 0:
                predicted_me = 0
            
            # Alert logic
            # alert_me = False
            # if actual_fuel is not None and actual_fuel > 0:
            #     percentage_diff = abs(predicted_me - actual_fuel) / actual_fuel * 100
                
            #     if actual_fuel < 50:
            #         threshold = 10.0
            #     elif actual_fuel < 150:
            #         threshold = 6.0
            #     else:
            #         threshold = 4.0
                
            #     alert_me = percentage_diff > threshold
            # Alert logic - ABSOLUTE DIFFERENCE ONLY
            
            alert_me = False
            if actual_fuel is not None and actual_fuel > 0:
                absolute_diff = abs(predicted_me - actual_fuel)
                
                # Simple: Only alert if difference is genuinely large
                if actual_fuel < 100:
                    alert_threshold = 100.0  # Need 100+ kg/h difference to care
                elif actual_fuel < 300:
                    alert_threshold = 150.0  # Need 150+ kg/h difference
                else:
                    alert_threshold = 200.0  # Need 200+ kg/h difference
                
                alert_me = absolute_diff > alert_threshold

        
        # AE Prediction
        predicted_ae = None
        alert_ae = None
        
        if ae_available and all(x is not None for x in [ae1_power, ae2_power, ae3_power, ae4_power]):
            ae_input = pd.DataFrame({
                'AE1_POW_act_kW@AVG': [ae1_power],
                'AE2_POW_act_kW@AVG': [ae2_power],
                'AE3_POW_act_kW@AVG': [ae3_power],
                'AE4_POW_act_kW@AVG': [ae4_power]
            })
            
            X_scaled = ae_scaler_X.transform(ae_input)
            X_pca = ae_pca_X.transform(X_scaled)
            X_tensor = torch.tensor(X_pca, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                y_pred_scaled = ae_model(X_tensor).cpu().numpy().flatten()
            
            predicted_ae = round(float(ae_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]), 2)
            
            if predicted_ae < 0:
                predicted_ae = 0
            
            # Alert logic
            alert_ae = False
            if actual_ae_total_fuel > 0:
                absolute_diff = abs(predicted_ae - actual_ae_total_fuel)
                percentage_diff = absolute_diff / actual_ae_total_fuel * 100
                
                # Alert if BOTH conditions met
                if percentage_diff > 8.0 and absolute_diff > 15:
                    alert_ae = True
        
        result = {
            'predicted_me': predicted_me,
            'predicted_ae': predicted_ae,
            'alert_ae': alert_ae,
            'alert_me': alert_me
        }
        
        print("RESPP", result)
        return result
        
    except Exception as e:
        print("Error", e)
        return JSONResponse(
            content={
                "error": str(e),
                'predicted_me': None,
                'predicted_ae': None,
                'alert_me': None,
                'alert_ae': None
            },
            status_code=200
        )

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

# ============= PREDICTIVE MAINTENANCE =============

# @app.post("/vessels/{imo}/maintenance/predict")
@app.post("/predict")
async def predict_maintenance():
    """Run predictive maintenance analysis for specific vessel"""
    try:
        logger.info(f"Starting predictive maintenance analysis for vessel")
        
        
        # Step 1: Get fault matrix from external API
        fault_matrix_url = "https://cm.memphis-marine.com/api/faultsense/get/?fk_vessel=1&tag1=me&tag2=fm"
        response = requests.get(fault_matrix_url)
        
        if response.status_code != 200:
            return {"status": "error", "message": "Failed to fetch fault matrix"}
        
        api_data = response.json()
        
        if not api_data.get('success'):
            return {"status": "error", "message": "API returned unsuccessful response"}
        
        # Extract fault matrix data and ID
        fault_matrix_id = api_data['data']['id']
        fault_matrix_data = api_data['data']['data']
        
        # Step 2: Create config with fault matrix
        user_config = {
            'fault_matrix': fault_matrix_data
        }
        
        # Step 3: Run pipeline using imported functions
        fixed_data_path = r'C:\Users\User\Desktop\FaultSenseAI_source\smartmaintanace\mgd.csv'
        result = run_pipeline(fixed_data_path, user_config)

        # Step 4: Process results
        today_str = datetime.now().strftime("%Y-%m-%d")
        pickle_path = f'complete_maintenance_results_{today_str}.pickle'
        print(f"Looking for pickle file: {pickle_path}")
        print(f"Pickle file exists: {os.path.exists(pickle_path)}")

        if os.path.exists(pickle_path):
            print("Pickle file found, starting processing...")
            utility_dict_test = load_config_with_overrides(user_config)
            print("About to call process_smart_maintenance_results...")
            maintenance_result = process_smart_maintenance_results(pickle_path, utility_dict_test)
            print(f"process_smart_maintenance_results returned: {maintenance_result}")
            print("process_smart_maintenance_results completed")
            
            if maintenance_result.get('totalRecords', 0) > 0:

            # Step 5: Post results to external API
                result_payload = {
                    "status": 200,
                    "fk_faultsenseconfig": fault_matrix_id,
                    "data": maintenance_result
                }
            
            # Post to external API
                post_url = "https://cm.memphis-marine.com/api/faultsense/alert/"
                post_response = requests.post(post_url, json=result_payload)
                print(post_response)
                if post_response.status_code == 200:
                    return {
                        "status": 200,
                        "message": "Prediction completed and results posted successfully",
                        "fault_matrix_id": fault_matrix_id,
                        # "vessel_imo": imo,
                        "data": maintenance_result
                }
                else:
                    return {
                        "status": 206,
                        "message": "Prediction completed but failed to post results",
                        "fault_matrix_id": fault_matrix_id,
                        # "vessel_imo": imo,
                        "data": maintenance_result,
                        "post_error": f"Failed to post results: {post_response.status_code}"
                    }
        else:
            return {"status": "error", "message": "Maintenance results not generated"}
        
    except requests.RequestException as e:
        logger.error(f"Network error for vessel ")
        return {"status": "error", "message": f"Network error: {str(e)}"}
    except Exception as e:
        logger.error(f"Error predicting maintenance for vessel ")
        return {"status": "error", "message": str(e)}


# ============= STARTUP =============
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI Marine Engineering AI System...")
    uvicorn.run(app, host="0.0.0.0", port=5004)