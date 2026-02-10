# document_processor.py
import docx
import pandas as pd
from typing import List, Dict
import re

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_word_document(file_path: str) -> List[Dict]:
    """Extract troubleshooting info from Word document"""
    doc = docx.Document(file_path)
    chunks = []
    
    current_equipment = ""
    current_issue = ""
    current_solution = ""
    
    for para in doc.paragraphs:
        text = clean_text(para.text)
        if not text:
            continue
        
        # Detect headers (bold or style)
        if para.style.name.startswith('Heading') or (para.runs and para.runs[0].bold):
            # New equipment/section
            if current_equipment and current_solution:
                chunks.append({
                    "equipment": current_equipment,
                    "issue": current_issue,
                    "solution": current_solution,
                    "text": f"Equipment: {current_equipment}\nIssue: {current_issue}\nSolution: {current_solution}"
                })
            current_equipment = text
            current_issue = ""
            current_solution = ""
        else:
            # Check if it's issue description
            if any(keyword in text.lower() for keyword in ['problem:', 'issue:', 'symptom:', 'alarm:']):
                current_issue = text
            # Check if it's solution
            elif any(keyword in text.lower() for keyword in ['solution:', 'action:', 'fix:', 'troubleshooting:']):
                current_solution = text
            else:
                # Append to current section
                if current_issue:
                    current_solution += " " + text
                else:
                    current_issue += " " + text
    
    # Add last entry
    if current_equipment and current_solution:
        chunks.append({
            "equipment": current_equipment,
            "issue": current_issue,
            "solution": current_solution,
            "text": f"Equipment: {current_equipment}\nIssue: {current_issue}\nSolution: {current_solution}"
        })
    
    return chunks

def process_excel_document(file_path: str) -> List[Dict]:
    """Extract troubleshooting info from Excel"""
    df = pd.read_excel(file_path)
    chunks = []
    
    # Detect column names flexibly
    equipment_col = None
    issue_col = None
    solution_col = None
    parts_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'equipment' in col_lower or 'system' in col_lower:
            equipment_col = col
        elif 'issue' in col_lower or 'problem' in col_lower or 'symptom' in col_lower:
            issue_col = col
        elif 'solution' in col_lower or 'action' in col_lower or 'fix' in col_lower:
            solution_col = col
        elif 'part' in col_lower or 'check' in col_lower or 'suspect' in col_lower:
            parts_col = col
    
    if not all([equipment_col, issue_col, solution_col]):
        raise ValueError(f"Excel must have columns: Equipment, Issue, Solution. Found: {list(df.columns)}")
    
    for _, row in df.iterrows():
        equipment = clean_text(str(row[equipment_col]))
        issue = clean_text(str(row[issue_col]))
        solution = clean_text(str(row[solution_col]))
        parts = clean_text(str(row[parts_col])) if parts_col else ""
        
        if equipment and solution:
            # Build searchable text
            text = f"Equipment: {equipment}\nIssue: {issue}\nSolution: {solution}"
            if parts:
                text += f"\nSuspected Parts: {parts}"
            
            chunks.append({
                "equipment": equipment,
                "issue": issue,
                "solution": solution,
                "suspected_parts": parts,
                "text": text
            })
    
    return chunks

def process_document(file_path: str) -> List[Dict]:
    """Process Word or Excel document"""
    if file_path.endswith('.docx'):
        return process_word_document(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return process_excel_document(file_path)
    else:
        raise ValueError("Unsupported file format. Use .docx or .xlsx")