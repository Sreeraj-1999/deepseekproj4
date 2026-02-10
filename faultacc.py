# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Dict, Optional
# import pymongo
# import pandas as pd
# import numpy as np
# import math
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # MongoDB connection
# MONGO_URI = "mongodb://admin:secret@192.168.17.21:27017/?authSource=admin"
# client = pymongo.MongoClient(MONGO_URI)
# db = client['bsm']  # Your DB name

# # Pydantic models
# class DataModel(BaseModel):
#     tag_name: str

# class InputModel(BaseModel):
#     success: bool
#     data: DataModel
#     metadata: str  # This is the IMO number

# @app.post("/tag/correlation/")
# async def tag_correlation(input_json: InputModel):
#     logger.info("🚀 Starting correlation analysis...")
    
#     imo_number = input_json.metadata
#     tag_name = input_json.data.tag_name
#     logger.info(f"📊 Input received - IMO: {imo_number}, Tag: {tag_name}")

#     # Validate inputs
#     logger.info("✅ Step 1: Validating inputs...")
#     if not imo_number or not isinstance(imo_number, str):
#         logger.error("❌ Invalid IMO number")
#         raise HTTPException(status_code=400, detail="Invalid IMO number")
#     logger.info("✅ Input validation passed")

#     # Select the collection dynamically based on IMO
#     logger.info(f"✅ Step 2: Connecting to collection 'imo_{imo_number}'...")
#     collection = db[f"imo_{imo_number}"]

#     # Fetch all records for this vessel
#     logger.info("✅ Step 3: Fetching vessel data from MongoDB...")
#     vessel_data = list(collection.find())
#     logger.info(f"✅ Fetched {len(vessel_data)} records from database")
    
#     if not vessel_data:
#         logger.error(f"❌ No data found for vessel IMO {imo_number}")
#         raise HTTPException(status_code=404, detail=f"No data found for vessel IMO {imo_number}")

#     # Convert to DataFrame
#     logger.info("✅ Step 4: Converting to DataFrame...")
#     df = pd.DataFrame(vessel_data)
#     logger.info(f"✅ DataFrame created with shape: {df.shape}")
#     logger.info(f"✅ Available columns: {list(df.columns)}")
    
#     # Remove MongoDB's _id field if present
#     if '_id' in df.columns:
#         logger.info("✅ Step 5: Removing MongoDB '_id' field...")
#         df = df.drop('_id', axis=1)
#         logger.info("✅ '_id' field removed")
#     else:
#         logger.info("✅ Step 5: No '_id' field found to remove")

#     # Check if target tag exists
#     logger.info(f"✅ Step 6: Checking if tag '{tag_name}' exists...")
#     if tag_name not in df.columns:
#         logger.error(f"❌ Tag {tag_name} not found in vessel data")
#         raise HTTPException(status_code=404, detail=f"Tag {tag_name} not found in vessel data")
#     logger.info(f"✅ Target tag '{tag_name}' found")

#     # Ensure the target tag is numeric
#     logger.info(f"✅ Step 7: Checking if tag '{tag_name}' is numeric...")
#     if not pd.api.types.is_numeric_dtype(df[tag_name]):
#         logger.error(f"❌ Tag {tag_name} is not numeric. Type: {df[tag_name].dtype}")
#         raise HTTPException(status_code=400, detail=f"Tag {tag_name} is not numeric")
#     logger.info(f"✅ Target tag is numeric (type: {df[tag_name].dtype})")

#     # Find numeric columns
#     logger.info("✅ Step 8: Identifying numeric columns...")
#     numeric_columns = [col for col in df.columns if col != tag_name and pd.api.types.is_numeric_dtype(df[col])]
#     logger.info(f"✅ Found {len(numeric_columns)} numeric columns for analysis: {numeric_columns}")

#     # Compute correlations with numeric columns
#     logger.info("✅ Step 9: Starting correlation analysis...")
#     correlation_results = {}
#     skipped_columns = []
#     processed_columns = []
    
#     for i, col in enumerate(numeric_columns, 1):
#         logger.info(f"📈 Analyzing column {i}/{len(numeric_columns)}: '{col}'")
        
#         # Get data for both columns, dropping NaN values
#         original_count = len(df)
#         valid_data = df[[tag_name, col]].dropna()
#         valid_count = len(valid_data)
#         logger.info(f"   📊 Data points: {valid_count}/{original_count} (after removing NaN)")
        
#         # Need at least 2 data points for correlation
#         if valid_count < 2:
#             logger.warning(f"   ⚠️  Skipping '{col}': insufficient data points ({valid_count} < 2)")
#             skipped_columns.append(f"{col}: insufficient data")
#             continue
            
#         # Check variance
#         tag_variance = valid_data[tag_name].var()
#         col_variance = valid_data[col].var()
#         logger.info(f"   📊 Variance - {tag_name}: {tag_variance:.6f}, {col}: {col_variance:.6f}")
        
#         # Check if either column has zero variance (all values the same)
#         if tag_variance == 0 or col_variance == 0:
#             logger.warning(f"   ⚠️  Skipping '{col}': zero variance detected")
#             skipped_columns.append(f"{col}: zero variance")
#             continue
            
#         try:
#             # Calculate correlation
#             logger.info(f"   🔄 Calculating correlation...")
#             corr = valid_data[tag_name].corr(valid_data[col])
#             logger.info(f"   📈 Raw correlation: {corr}")
            
#             # Check if correlation is valid (not NaN or infinite)
#             if pd.isna(corr):
#                 logger.warning(f"   ⚠️  Skipping '{col}': correlation is NaN")
#                 skipped_columns.append(f"{col}: NaN correlation")
#                 continue
                
#             if math.isinf(corr):
#                 logger.warning(f"   ⚠️  Skipping '{col}': correlation is infinite")
#                 skipped_columns.append(f"{col}: infinite correlation")
#                 continue
                
#             # Convert to percentage and round
#             corr_percentage = round(float(corr * 100), 2)
#             correlation_results[col] = corr_percentage
#             processed_columns.append(col)
#             logger.info(f"   ✅ Successfully calculated: {corr_percentage}%")
            
#         except Exception as e:
#             logger.error(f"   ❌ Error calculating correlation for '{col}': {e}")
#             skipped_columns.append(f"{col}: calculation error - {str(e)}")
#             continue

#     # Final summary
#     logger.info("✅ Step 10: Correlation analysis completed!")
#     logger.info(f"✅ Successfully processed: {len(processed_columns)} columns")
#     logger.info(f"⚠️  Skipped: {len(skipped_columns)} columns")
#     if skipped_columns:
#         logger.info(f"📝 Skipped columns details: {skipped_columns}")
    
#     logger.info(f"🎯 Final results: {len(correlation_results)} correlations calculated")

#     # Return results
#     return {
#         "success": True,
#         "tag_name": tag_name,
#         "imo_number": imo_number,
#         "correlations": correlation_results,
#         "total_records": len(df),
#         "analyzed_columns": len(numeric_columns),
#         "processed_columns": len(processed_columns),
#         "skipped_columns": len(skipped_columns),
#         "debug_info": {
#             "processed": processed_columns,
#             "skipped": skipped_columns
#         }
#     }
##############################################################
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Dict, Optional
# import pandas as pd
# import numpy as np
# import math
# import logging
# from sklearn.feature_selection import mutual_info_regression

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # Pydantic models
# class DataModel(BaseModel):
#     id: int
#     fk_equipmentmaster: int
#     fk_subequipmentmaster: int
#     fk_vessel: int
#     tag_flag: bool
#     name: Optional[str]
#     tag_name: str
#     lower_limit: float
#     upper_limit: float
#     aggregation: str
#     latex: Optional[str]

# class InputModel(BaseModel):
#     success: bool
#     data: DataModel
#     metadata: str  # This is the IMO number

# @app.post("/tag/correlation/")
# async def tag_correlation(input_json: InputModel):
#     logger.info("🚀 Starting mutual information analysis...")
    
#     imo_number = input_json.metadata
#     tag_name = input_json.data.tag_name
#     logger.info(f"📊 Input received - IMO: {imo_number}, Tag: {tag_name}")

#     # Validate inputs
#     logger.info("✅ Step 1: Validating inputs...")
#     if not imo_number or not isinstance(imo_number, str):
#         logger.error("❌ Invalid IMO number")
#         raise HTTPException(status_code=400, detail="Invalid IMO number")
    
#     if not tag_name or not isinstance(tag_name, str):
#         logger.error("❌ Invalid tag name")
#         raise HTTPException(status_code=400, detail="Invalid tag name")
#     logger.info("✅ Input validation passed")

#     # Read Excel file based on IMO number
#     logger.info(f"✅ Step 2: Reading Excel file for IMO {imo_number}...")
#     try:
#         excel_path = f"data/imo_{imo_number}.xlsx"  # Assuming files are stored like this
#         df = pd.read_excel(excel_path)
#         logger.info(f"✅ Excel file loaded with shape: {df.shape}")
#         logger.info(f"✅ Available columns: {list(df.columns)}")
#     except Exception as e:
#         logger.error(f"❌ Error reading Excel file: {e}")
#         raise HTTPException(status_code=400, detail=f"Error reading Excel file for IMO {imo_number}: {str(e)}")

#     # Check if target tag exists
#     logger.info(f"✅ Step 3: Checking if tag '{tag_name}' exists...")
#     if tag_name not in df.columns:
#         logger.error(f"❌ Tag {tag_name} not found in Excel data")
#         raise HTTPException(status_code=404, detail=f"Tag {tag_name} not found in vessel data")
#     logger.info(f"✅ Target tag '{tag_name}' found")

#     # Ensure the target tag is numeric
#     logger.info(f"✅ Step 4: Checking if tag '{tag_name}' is numeric...")
#     if not pd.api.types.is_numeric_dtype(df[tag_name]):
#         logger.error(f"❌ Tag {tag_name} is not numeric. Type: {df[tag_name].dtype}")
#         raise HTTPException(status_code=400, detail=f"Tag {tag_name} is not numeric")
#     logger.info(f"✅ Target tag is numeric (type: {df[tag_name].dtype})")

#     # Find numeric columns
#     logger.info("✅ Step 5: Identifying numeric columns...")
#     numeric_columns = [col for col in df.columns if col != tag_name and pd.api.types.is_numeric_dtype(df[col])]
#     logger.info(f"✅ Found {len(numeric_columns)} numeric columns for analysis: {numeric_columns}")

#     # Compute mutual information with numeric columns
#     logger.info("✅ Step 6: Starting mutual information analysis...")
#     mutual_info_results = {}
#     skipped_columns = []
#     processed_columns = []
    
#     for i, col in enumerate(numeric_columns, 1):
#         logger.info(f"📈 Analyzing column {i}/{len(numeric_columns)}: '{col}'")
        
#         # Get data for both columns, dropping NaN values
#         original_count = len(df)
#         valid_data = df[[tag_name, col]].dropna()
#         valid_count = len(valid_data)
#         logger.info(f"   📊 Data points: {valid_count}/{original_count} (after removing NaN)")
        
#         # Need at least 2 data points for mutual information
#         if valid_count < 2:
#             logger.warning(f"   ⚠️  Skipping '{col}': insufficient data points ({valid_count} < 2)")
#             skipped_columns.append(f"{col}: insufficient data")
#             continue
            
#         # Check variance
#         tag_variance = valid_data[tag_name].var()
#         col_variance = valid_data[col].var()
#         logger.info(f"   📊 Variance - {tag_name}: {tag_variance:.6f}, {col}: {col_variance:.6f}")
        
#         # If target tag has zero variance, skip
#         if tag_variance == 0:
#             logger.warning(f"   ⚠️  Skipping '{col}': target tag has zero variance")
#             skipped_columns.append(f"{col}: target zero variance")
#             continue
            
#         try:
#             # Prepare data for mutual information calculation
#             logger.info(f"   🔄 Calculating mutual information...")
#             X = valid_data[[col]].values  # Features (2D array required)
#             y = valid_data[tag_name].values  # Target (1D array)
            
#             # Calculate mutual information
#             mi_score = mutual_info_regression(X, y, random_state=42)
#             mi_value = mi_score[0]  # mutual_info_regression returns array
            
#             logger.info(f"   📈 Raw MI score: {mi_value:.6f}")
            
#             # Check if MI is valid (not NaN or infinite)
#             if pd.isna(mi_value):
#                 logger.warning(f"   ⚠️  Skipping '{col}': MI is NaN")
#                 skipped_columns.append(f"{col}: NaN MI")
#                 continue
                
#             if math.isinf(mi_value):
#                 logger.warning(f"   ⚠️  Skipping '{col}': MI is infinite")
#                 skipped_columns.append(f"{col}: infinite MI")
#                 continue
                
#             # Convert to percentage (multiply by 100) and round
#             mi_percentage = round(float(mi_value * 100), 2)
#             mutual_info_results[col] = mi_percentage
#             processed_columns.append(col)
#             logger.info(f"   ✅ Successfully calculated: {mi_percentage}%")
            
#         except Exception as e:
#             logger.error(f"   ❌ Error calculating MI for '{col}': {e}")
#             skipped_columns.append(f"{col}: calculation error - {str(e)}")
#             continue

#     # Final summary
#     logger.info("✅ Step 7: Mutual information analysis completed!")
#     logger.info(f"✅ Successfully processed: {len(processed_columns)} columns")
#     logger.info(f"⚠️  Skipped: {len(skipped_columns)} columns")
#     if skipped_columns:
#         logger.info(f"📝 Skipped columns details: {skipped_columns}")
    
#     logger.info(f"🎯 Final results: {len(mutual_info_results)} mutual information scores calculated")

#     # Return results in your exact format
#     return {
#         "success": True,
#         "tag_name": tag_name,
#         "imo_number": imo_number,
#         "correlations": mutual_info_results
#     }

###################################

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import math
import logging
from sklearn.feature_selection import mutual_info_regression
import asyncpg
import asyncio
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_db_connection():
    """Create database connection"""
    try:
        # Use URL-encoded password: @ becomes %40, $ becomes %24
        DATABASE_URL = "postgresql://memphis:Memphis%401234%24@192.168.18.176/OBANEXT5"
        logger.info(f"🔗 Attempting to connect to database at 192.168.18.176...")
        
        conn = await asyncio.wait_for(
            asyncpg.connect(DATABASE_URL), 
            timeout=10.0
        )
        logger.info("✅ Database connection successful!")
        return conn
    except asyncio.TimeoutError:
        logger.error("❌ Database connection timeout")
        raise HTTPException(status_code=500, detail="Database connection timeout")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

app = FastAPI()

# Pydantic models
class DataModel(BaseModel):
    id: int
    fk_equipmentmaster: int
    fk_subequipmentmaster: int
    fk_vessel: int
    tag_flag: bool
    name: Optional[str]
    tag_name: Optional[str]
    lower_limit: float
    upper_limit: float
    aggregation: str
    latex: Optional[str]

class InputModel(BaseModel):
    success: bool
    data: List[DataModel]  # ARRAY of DataModel objects
    metadata: str  # This is the IMO number

@app.post("/tag/correlation/")
async def tag_correlation(input_json: InputModel):
    logger.info("🚀 Starting mutual information analysis...")
    
    imo_number = input_json.metadata
    data_items = input_json.data
    logger.info(f"📊 Input received - IMO: {imo_number}, Number of items: {len(data_items)}")

    # Validate inputs
    logger.info("✅ Step 1: Validating inputs...")
    if not imo_number or not isinstance(imo_number, str):
        logger.error("❌ Invalid IMO number")
        raise HTTPException(status_code=400, detail="Invalid IMO number")
    
    if not data_items or len(data_items) == 0:
        logger.error("❌ No data items provided")
        raise HTTPException(status_code=400, detail="No data items provided")
    logger.info("✅ Input validation passed")

    # Read Excel file based on IMO number
    logger.info(f"✅ Step 2: Reading Excel file for IMO {imo_number}...")
    try:
        # excel_path = f"data/imo_{imo_number}.xlsx"
        excel_path=r"C:\Users\User\Desktop\siemens\freya_schulte\imo_9665671_ME1_FMS_act_kgPh@AVG_dump.csv"
        df = pd.read_csv(excel_path)
        logger.info(f"✅ Excel file loaded with shape: {df.shape}")
        # logger.info(f"✅ Available columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"❌ Error reading Excel file: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading Excel file for IMO {imo_number}: {str(e)}")

    # Process each data item to create calculated columns if needed
    logger.info(f"✅ Step 3: Processing {len(data_items)} data items...")
    target_tags = []
    
    for i, item in enumerate(data_items, 1):
        logger.info(f"🔄 Processing item {i}/{len(data_items)}: ID {item.id}")
        
        # Check if it's a calculated column
        is_calculated_column = item.tag_name is None and item.name and item.latex
        
        if is_calculated_column:
            logger.info(f"   📊 Creating calculated column '{item.name}' from LaTeX...")
            try:
                # Parse LaTeX equation to extract tag names
                import re
                
                # Extract tag names from LaTeX (look for \text{tag_name} pattern)
                tag_pattern = r'\\text\{([^}]+)\}'
                found_tags = re.findall(tag_pattern, item.latex)
                logger.info(f"   📋 Found tags in equation: {found_tags}")
                
                # Check if all referenced tags exist in DataFrame
                missing_tags = [tag for tag in found_tags if tag not in df.columns]
                if missing_tags:
                    logger.error(f"❌ Missing tags in Excel data for item {item.id}: {missing_tags}")
                    raise HTTPException(status_code=404, detail=f"Tags not found in vessel data for item {item.id}: {missing_tags}")
                
                # Convert LaTeX to Python expression
                python_expression = item.latex
                
                # Replace LaTeX syntax with Python syntax
                python_expression = python_expression.replace('\\left(', '(')
                python_expression = python_expression.replace('\\right)', ')')
                python_expression = python_expression.replace('\\ ', ' ')
                python_expression = python_expression.replace('\\div', '/')
                python_expression = python_expression.replace('\\times', '*')
                python_expression = python_expression.replace('^', '**')
                
                # Replace \text{tag_name} with df['tag_name']
                for tag in found_tags:
                    python_expression = python_expression.replace(f'\\text{{{tag}}}', f"df['{tag}']")
                
                logger.info(f"   🔄 Python expression: {python_expression}")
                
                # Calculate the new column
                df[item.name] = eval(python_expression)
                logger.info(f"   ✅ Calculated column '{item.name}' created successfully")
                
                # Add to target tags list
                target_tags.append(item.name)
                
            except Exception as e:
                logger.error(f"❌ Error creating calculated column for item {item.id}: {e}")
                raise HTTPException(status_code=400, detail=f"Error creating calculated column for item {item.id}: {str(e)}")
        
        else:
            # Regular existing tag
            if not item.tag_name:
                logger.error(f"❌ Item {item.id}: Neither tag_name nor valid name/latex provided")
                raise HTTPException(status_code=400, detail=f"Item {item.id}: Neither tag_name nor valid name/latex provided")
            
            logger.info(f"   📊 Using existing tag '{item.tag_name}'")
            
            # Check if tag exists in DataFrame
            if item.tag_name not in df.columns:
                logger.error(f"❌ Tag {item.tag_name} not found in Excel data for item {item.id}")
                raise HTTPException(status_code=404, detail=f"Tag {item.tag_name} not found in vessel data for item {item.id}")
            
            # Add to target tags list
            target_tags.append(item.tag_name)
    
    logger.info(f"✅ Step 3 completed. Target tags for analysis: {target_tags}")

    # Now run MI analysis for each target tag
    logger.info("✅ Step 4: Starting mutual information analysis for all target tags...")
    all_results = {}
    
    for tag_index, target_tag in enumerate(target_tags, 1):
        logger.info(f"🎯 Analyzing target tag {tag_index}/{len(target_tags)}: '{target_tag}'")
        
        # Ensure the target tag is numeric
        if not pd.api.types.is_numeric_dtype(df[target_tag]):
            logger.error(f"❌ Tag {target_tag} is not numeric. Type: {df[target_tag].dtype}")
            raise HTTPException(status_code=400, detail=f"Tag {target_tag} is not numeric")
        
        # Find numeric columns (excluding the current target tag and other target tags)
        other_targets = [t for t in target_tags if t != target_tag]
        numeric_columns = [col for col in df.columns 
                          if col != target_tag 
                          and pd.api.types.is_numeric_dtype(df[col])]
        
        logger.info(f"   📊 Found {len(numeric_columns)} numeric columns for analysis")
        logger.info(f"   🔍 Pre-filtering {len(numeric_columns)} columns...")
        useful_columns = []
        for col in numeric_columns:
            col_valid_count = df[col].notna().sum()
            col_variance = df[col].var()
            if col_valid_count >= 2 and col_variance > 1e-10:
                useful_columns.append(col)

        logger.info(f"   ✅ Reduced to {len(useful_columns)} useful columns")
        numeric_columns = useful_columns  # Replace the original list

        
        # Compute mutual information with numeric columns
        mutual_info_results = {}
        skipped_columns = []
        processed_columns = []
        
        for i, col in enumerate(numeric_columns, 1):
            # logger.info(f"   📈 Analyzing column {i}/{len(numeric_columns)}: '{col}'")
            
            # Get data for both columns, dropping NaN values
            original_count = len(df)
            valid_data = df[[target_tag, col]].dropna()
            valid_count = len(valid_data)
            
            # Need at least 2 data points for mutual information
            if valid_count < 2:
                logger.warning(f"      ⚠️  Skipping '{col}': insufficient data points ({valid_count} < 2)")
                skipped_columns.append(f"{col}: insufficient data")
                continue
                
            # Check variance
            target_variance = valid_data[target_tag].var()
            col_variance = valid_data[col].var()
            
            # If target tag has zero variance, skip
            if target_variance == 0 or col_variance == 0:
                # logger.warning(f"      ⚠️  Skipping '{col}': target tag has zero variance")
                skipped_columns.append(f"{col}: target zero variance")
                continue
                
            try:
                # Prepare data for mutual information calculation
                X = valid_data[[col]].values  # Features (2D array required)
                y = valid_data[target_tag].values  # Target (1D array)
                
                # Calculate mutual information
                mi_score = mutual_info_regression(X, y, random_state=42)
                mi_value = mi_score[0]  # mutual_info_regression returns array
                
                # Check if MI is valid (not NaN or infinite)
                if pd.isna(mi_value) or math.isinf(mi_value):
                    logger.warning(f"      ⚠️  Skipping '{col}': invalid MI value")
                    skipped_columns.append(f"{col}: invalid MI")
                    continue
                    
                # Convert to percentage (multiply by 100) and round
                # mi_percentage = round(float(mi_value * 100), 2)
                mi_normalized = mi_value / (mi_value + 1)  # Scales 0 to 1
                mi_percentage = round(float(mi_normalized * 100), 2)
                if mi_percentage < 5.0:  # Skip weak correlations
                    skipped_columns.append(f"{col}: weak correlation ({mi_percentage}%)")
                    continue
                mutual_info_results[col] = mi_percentage
                # sorted_results = dict(sorted(mutual_info_results.items(), key=lambda x: x[1], reverse=True)[:15])
                # mutual_info_results = sorted_results
                processed_columns.append(col)
                # logger.info(f"      ✅ Successfully calculated: {mi_percentage}%")
                
            except Exception as e:
                logger.error(f"      ❌ Error calculating MI for '{col}': {e}")
                skipped_columns.append(f"{col}: calculation error")
                continue
        
        mutual_info_results = dict(sorted(mutual_info_results.items(), key=lambda x: x[1], reverse=True)[:15])
        # Store results for this target tag
        all_results[target_tag] = {
            "correlations": mutual_info_results,
            "processed_columns": len(processed_columns),
            "skipped_columns": len(skipped_columns)
        }
        
        logger.info(f"   ✅ Completed analysis for '{target_tag}': {len(mutual_info_results)} correlations")

    # Final summary
    logger.info("✅ Step 5: All mutual information analysis completed!")
    logger.info(f"🎯 Analyzed {len(target_tags)} target tags")
    
    # Return results for all target tags
    # return {
    #     "success": True,
    #     "imo_number": imo_number,
    #     "total_items": len(data_items),
    #     "target_tags": target_tags,
    #     "results": all_results
    # }
    logger.info("✅ Step 6: Saving results to database...")
    try:
        conn = await get_db_connection()
        
        for i, item in enumerate(data_items):
            target_tag = target_tags[i]
            result = all_results[target_tag]
            
            # Transform correlations to the required format
            correlations_list = [
                {
                    "tag": tag_name,
                    "percentage": percentage,
                    "flag": False  # All flags are False as requested
                }
                for tag_name, percentage in result["correlations"].items()
            ]
            
            # Create MlResponse JSON
            ml_response = {
                target_tag: {
                    "correlations": correlations_list,
                    "processed_columns": result["processed_columns"],
                    "skipped_columns": result["skipped_columns"]
                }
            }
            
            # Insert into database - using item.id as fk_efdmaster
            logger.info(f"   DEBUG: item.id = {item.id}")
            logger.info(f"   DEBUG: target_tag = {target_tag}")
            logger.info(f"   DEBUG: ml_response keys = {list(ml_response.keys())}")
            await conn.execute("""
                INSERT INTO "ImportantFeaturesMaster" ("fk_efdmaster", "MlResponse")
                VALUES ($1, $2)
            """, item.id, json.dumps(ml_response))
            
            logger.info(f"   💾 Saved results for item {item.id} ({target_tag})")
        
        await conn.close()
        logger.info("✅ All results saved to database successfully!")
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Return success response
    return {
        "success": True,
        "imo_number": imo_number,
        "total_items": len(data_items),
        "target_tags": target_tags,
        "message": "Results saved to database successfully"
    }