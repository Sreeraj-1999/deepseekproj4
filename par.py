# import os
# import re
# from pathlib import Path
# import pdfplumber
# from unstructured.partition.pdf import partition_pdf
# from unstructured.chunking.title import chunk_by_title
# import logging
# from typing import List, Dict, Tuple
# from datetime import datetime

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def clean_text_content(text: str) -> str:
#     """Clean and normalize text content"""
#     if not text:
#         return ""
    
#     # Remove script/style blocks
#     text = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL|re.IGNORECASE)
#     # Remove HTML tags
#     text = re.sub(r'<[^>]+>', '', text)
#     # Remove LaTeX markers
#     text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
#     text = re.sub(r'\\\[(.*?)\\\]', r'\1', text)
#     # Remove escape characters
#     text = text.replace("\\", "")
#     # Collapse multiple spaces/newlines
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     # Preserve bullet points
#     text = text.replace(" •", "\n•")
#     text = text.replace(" -", "\n-")
#     text = text.replace(" *", "\n*")
    
#     return text.strip()

# def get_element_text(el) -> str:
#     """Extract text from Unstructured element"""
#     try:
#         if hasattr(el, "get_text"):
#             return el.get_text() or ""
#         if hasattr(el, "text"):
#             return el.text or ""
#         return str(el) or ""
#     except Exception:
#         try:
#             return str(el)
#         except Exception:
#             return ""

# def get_element_page(el) -> int:
#     """Extract page number from element metadata"""
#     try:
#         meta = None
#         if hasattr(el, "metadata"):
#             meta = getattr(el, "metadata")
#         elif hasattr(el, "meta"):
#             meta = getattr(el, "meta")
#         if isinstance(meta, dict):
#             return int(meta.get("page_number") or meta.get("page") or 1)
#         if meta and hasattr(meta, "page_number"):
#             return int(getattr(meta, "page_number", 1))
#     except Exception:
#         pass
#     return 1

# def get_element_category(el) -> str:
#     """Get element category/type"""
#     try:
#         return getattr(el, 'category', '') or ''
#     except Exception:
#         return ''

# def extract_text_with_unstructured(pdf_path: str) -> Tuple[List[Dict], bool]:
#     """Extract text using Unstructured with fallback strategies"""
#     elements = []
    
#     for strategy in ["hi_res", "fast", "auto"]:
#         try:
#             logger.info(f"Trying Unstructured strategy: {strategy}")
#             elements = partition_pdf(
#                 filename=pdf_path,
#                 strategy=strategy,
#                 infer_table_structure=False,  # We'll use pdfplumber for tables
#                 include_metadata=True,
#             )
#             if elements:
#                 logger.info(f"Success with strategy: {strategy}")
#                 break
#         except Exception as e:
#             logger.warning(f"Strategy '{strategy}' failed: {str(e)}")
#             continue
    
#     if not elements:
#         logger.error("All Unstructured strategies failed")
#         return [], False
    
#     # Process elements
#     processed_elements = []
#     for el in elements:
#         category = get_element_category(el)
#         # Skip non-text elements but keep structure
#         if category.lower() in ("image", "figurecaption", "pagebreak"):
#             continue
            
#         text = get_element_text(el)
#         page = get_element_page(el)
        
#         if text and text.strip():
#             processed_elements.append({
#                 'text': clean_text_content(text),
#                 'page': page,
#                 'category': category,
#                 'element_type': 'text'
#             })
    
#     return processed_elements, True

# def extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict]:
#     """Extract tables using pdfplumber"""
#     table_elements = []
    
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page_num, page in enumerate(pdf.pages, 1):
#                 try:
#                     tables = page.extract_tables() or []
#                     if not tables:
#                         continue
                        
#                     logger.info(f"Page {page_num}: found {len(tables)} tables")
                    
#                     for table_idx, table in enumerate(tables):
#                         try:
#                             if not table or len(table) == 0:
#                                 continue
                                
#                             # Format table as structured text
#                             table_text = format_table(table, page_num, table_idx)
                            
#                             if table_text:
#                                 table_elements.append({
#                                     'text': clean_text_content(table_text),
#                                     'page': page_num,
#                                     'category': 'Table',
#                                     'element_type': 'table',
#                                     'table_index': table_idx
#                                 })
                                
#                                 # Create individual row entries for better searchability
#                                 if len(table) > 1:  # Has headers + data
#                                     headers = table[0]
#                                     for row_idx, row in enumerate(table[1:], 1):
#                                         row_text = format_table_row(headers, row, page_num, table_idx, row_idx)
#                                         if row_text:
#                                             table_elements.append({
#                                                 'text': clean_text_content(row_text),
#                                                 'page': page_num,
#                                                 'category': 'TableRow',
#                                                 'element_type': 'table_row',
#                                                 'table_index': table_idx,
#                                                 'row_index': row_idx
#                                             })
                                            
#                         except Exception as e:
#                             logger.warning(f"Failed to process table {table_idx} on page {page_num}: {str(e)}")
                            
#                 except Exception as e:
#                     logger.warning(f"Table extraction error on page {page_num}: {str(e)}")
                    
#     except Exception as e:
#         logger.error(f"pdfplumber failed for {pdf_path}: {str(e)}")
    
#     return table_elements

# def format_table(table: List[List], page_num: int, table_idx: int) -> str:
#     """Format table for text output"""
#     if not table or len(table) == 0:
#         return ""
    
#     formatted_lines = []
#     formatted_lines.append(f"TABLE {table_idx + 1} (Page {page_num}):")
#     formatted_lines.append("=" * 50)
    
#     headers = table[0] if table else []
    
#     for row_idx, row in enumerate(table):
#         if row_idx == 0:
#             formatted_lines.append("HEADERS: " + " | ".join(str(cell or "").strip() for cell in row))
#             formatted_lines.append("-" * 50)
#         else:
#             row_data = []
#             for col_idx, cell in enumerate(row):
#                 if col_idx < len(headers) and headers[col_idx]:
#                     header = str(headers[col_idx]).strip()
#                     value = str(cell or "").strip()
#                     if header and value:
#                         row_data.append(f"{header}: {value}")
            
#             if row_data:
#                 formatted_lines.append(" | ".join(row_data))
    
#     return "\n".join(formatted_lines)

# def format_table_row(headers: List, row: List, page_num: int, table_idx: int, row_idx: int) -> str:
#     """Format individual table row"""
#     row_pairs = []
#     for col_idx, cell in enumerate(row):
#         if col_idx < len(headers) and headers[col_idx]:
#             header = str(headers[col_idx]).strip()
#             value = str(cell or "").strip()
#             if header and value:
#                 row_pairs.append(f"{header}: {value}")
    
#     if row_pairs:
#         return f"Table Row (Page {page_num}, Table {table_idx + 1}): " + " | ".join(row_pairs)
#     return ""

# def create_chunks(elements: List[Dict]) -> List[Dict]:
#     """Create chunks from elements using title-based chunking logic"""
#     chunks = []
#     current_chunk = []
#     current_title = ""
#     current_page = 1
#     chunk_id = 0
    
#     for element in elements:
#         category = element.get('category', '')
#         text = element.get('text', '')
#         page = element.get('page', 1)
        
#         # Check if this is a new section/title
#         if category in ['Title', 'Header'] or (len(text) < 100 and text.isupper()):
#             # Save previous chunk if it exists
#             if current_chunk:
#                 chunk_text = '\n'.join(current_chunk)
#                 if len(chunk_text.strip()) > 50:  # Only save substantial chunks
#                     chunks.append({
#                         'chunk_id': chunk_id,
#                         'title': current_title,
#                         'text': chunk_text.strip(),
#                         'page': current_page,
#                         'char_count': len(chunk_text),
#                         'element_count': len(current_chunk)
#                     })
#                     chunk_id += 1
            
#             # Start new chunk
#             current_chunk = [text]
#             current_title = text[:100] + "..." if len(text) > 100 else text
#             current_page = page
#         else:
#             # Add to current chunk
#             current_chunk.append(text)
            
#             # Split large chunks
#             current_text = '\n'.join(current_chunk)
#             if len(current_text) > 2000:
#                 chunks.append({
#                     'chunk_id': chunk_id,
#                     'title': current_title,
#                     'text': current_text.strip(),
#                     'page': current_page,
#                     'char_count': len(current_text),
#                     'element_count': len(current_chunk)
#                 })
#                 chunk_id += 1
#                 current_chunk = []
#                 current_title = ""
    
#     # Save final chunk
#     if current_chunk:
#         chunk_text = '\n'.join(current_chunk)
#         if len(chunk_text.strip()) > 50:
#             chunks.append({
#                 'chunk_id': chunk_id,
#                 'title': current_title,
#                 'text': chunk_text.strip(),
#                 'page': current_page,
#                 'char_count': len(chunk_text),
#                 'element_count': len(current_chunk)
#             })
    
#     return chunks

# def save_parsed_text(elements: List[Dict], output_path: str, pdf_name: str):
#     """Save parsed text to file"""
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(f"PARSED PDF: {pdf_name}\n")
#         f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write("=" * 80 + "\n\n")
        
#         current_page = 0
#         for element in elements:
#             page = element.get('page', 1)
#             category = element.get('category', 'Unknown')
#             text = element.get('text', '')
            
#             if page != current_page:
#                 f.write(f"\n{'='*20} PAGE {page} {'='*20}\n\n")
#                 current_page = page
            
#             f.write(f"[{category.upper()}]\n")
#             f.write(text)
#             f.write("\n\n" + "-" * 40 + "\n\n")

# def save_chunks(chunks: List[Dict], output_path: str, pdf_name: str):
#     """Save chunks to file"""
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(f"CHUNKS FROM: {pdf_name}\n")
#         f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Total Chunks: {len(chunks)}\n")
#         f.write("=" * 80 + "\n\n")
        
#         for chunk in chunks:
#             f.write(f"CHUNK #{chunk['chunk_id']}\n")
#             f.write(f"Title: {chunk['title']}\n")
#             f.write(f"Page: {chunk['page']}\n")
#             f.write(f"Characters: {chunk['char_count']}\n")
#             f.write(f"Elements: {chunk['element_count']}\n")
#             f.write("-" * 50 + "\n")
#             f.write(chunk['text'])
#             f.write("\n\n" + "=" * 80 + "\n\n")

# def parse_pdf(pdf_path: str):
#     """Main function to parse PDF and generate output files"""
#     pdf_path = Path(pdf_path)
    
#     if not pdf_path.exists():
#         raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
#     pdf_name = pdf_path.stem
#     output_dir = pdf_path.parent
    
#     # Output file paths
#     parsed_output = output_dir / f"{pdf_name}_parsed.txt"
#     chunks_output = output_dir / f"{pdf_name}_chunks.txt"
    
#     logger.info(f"Processing PDF: {pdf_path}")
    
#     # Extract text with Unstructured
#     text_elements, success = extract_text_with_unstructured(str(pdf_path))
    
#     if not success:
#         logger.error("Failed to extract text with Unstructured")
#         return
    
#     # Extract tables with pdfplumber
#     table_elements = extract_tables_with_pdfplumber(str(pdf_path))
    
#     # Combine all elements
#     all_elements = text_elements + table_elements
    
#     # Sort by page number
#     all_elements.sort(key=lambda x: x.get('page', 1))
    
#     logger.info(f"Extracted {len(text_elements)} text elements and {len(table_elements)} table elements")
    
#     # Save parsed text
#     save_parsed_text(all_elements, str(parsed_output), pdf_name)
#     logger.info(f"Saved parsed text to: {parsed_output}")
    
#     # Create and save chunks
#     chunks = create_chunks(all_elements)
#     save_chunks(chunks, str(chunks_output), pdf_name)
#     logger.info(f"Saved {len(chunks)} chunks to: {chunks_output}")
    
#     print(f"\n✓ Processing complete!")
#     print(f"✓ Parsed text saved to: {parsed_output}")
#     print(f"✓ Chunks saved to: {chunks_output}")
#     print(f"✓ Total elements processed: {len(all_elements)}")
#     print(f"✓ Total chunks created: {len(chunks)}")

# if __name__ == "__main__":
#     # HERE ADD PATH - Replace with your PDF file path
#     PDF_PATH = r"C:\4. Manuals\CD 01 Man B&W Diesel Engine\Book 2.pdf"
    
#     try:
#         parse_pdf(PDF_PATH)
#     except Exception as e:
#         logger.error(f"Error processing PDF: {str(e)}")
#         print(f"Error: {str(e)}")

import os
import re
from pathlib import Path
import pdfplumber
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import logging
from typing import List, Dict, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text_content(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove script/style blocks
    text = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL|re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove LaTeX markers
    text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
    text = re.sub(r'\\\[(.*?)\\\]', r'\1', text)
    # Remove escape characters
    text = text.replace("\\", "")
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Preserve bullet points
    text = text.replace(" •", "\n•")
    text = text.replace(" -", "\n-")
    text = text.replace(" *", "\n*")
    
    return text.strip()

def get_element_text(el) -> str:
    """Extract text from Unstructured element"""
    try:
        if hasattr(el, "get_text"):
            return el.get_text() or ""
        if hasattr(el, "text"):
            return el.text or ""
        return str(el) or ""
    except Exception:
        try:
            return str(el)
        except Exception:
            return ""

def get_element_page(el) -> int:
    """Extract page number from element metadata"""
    try:
        meta = None
        if hasattr(el, "metadata"):
            meta = getattr(el, "metadata")
        elif hasattr(el, "meta"):
            meta = getattr(el, "meta")
        if isinstance(meta, dict):
            return int(meta.get("page_number") or meta.get("page") or 1)
        if meta and hasattr(meta, "page_number"):
            return int(getattr(meta, "page_number", 1))
    except Exception:
        pass
    return 1

def get_element_category(el) -> str:
    """Get element category/type"""
    try:
        return getattr(el, 'category', '') or ''
    except Exception:
        return ''

def extract_text_with_unstructured(pdf_path: str) -> Tuple[List[Dict], bool]:
    """Extract text using Unstructured with fallback strategies"""
    elements = []
    
    for strategy in ["hi_res", "fast", "auto"]:
        try:
            logger.info(f"Trying Unstructured strategy: {strategy}")
            elements = partition_pdf(
                filename=pdf_path,
                strategy=strategy,
                infer_table_structure=False,  # We'll use pdfplumber for tables
                include_metadata=True,
            )
            if elements:
                logger.info(f"Success with strategy: {strategy}")
                break
        except Exception as e:
            logger.warning(f"Strategy '{strategy}' failed: {str(e)}")
            continue
    
    if not elements:
        logger.error("All Unstructured strategies failed")
        return [], False
    
    # Process elements
    processed_elements = []
    for el in elements:
        category = get_element_category(el)
        # Skip non-text elements but keep structure
        if category.lower() in ("image", "figurecaption", "pagebreak"):
            continue
            
        text = get_element_text(el)
        page = get_element_page(el)
        
        if text and text.strip():
            processed_elements.append({
                'text': clean_text_content(text),
                'page': page,
                'category': category,
                'element_type': 'text'
            })
    
    return processed_elements, True

def extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict]:
    """Extract tables using pdfplumber"""
    table_elements = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    tables = page.extract_tables() or []
                    if not tables:
                        continue
                        
                    logger.info(f"Page {page_num}: found {len(tables)} tables")
                    
                    for table_idx, table in enumerate(tables):
                        try:
                            if not table or len(table) == 0:
                                continue
                                
                            # Format table as structured text
                            table_text = format_table(table, page_num, table_idx)
                            
                            if table_text:
                                table_elements.append({
                                    'text': clean_text_content(table_text),
                                    'page': page_num,
                                    'category': 'Table',
                                    'element_type': 'table',
                                    'table_index': table_idx
                                })
                                
                                # Create individual row entries for better searchability
                                if len(table) > 1:  # Has headers + data
                                    headers = table[0]
                                    for row_idx, row in enumerate(table[1:], 1):
                                        row_text = format_table_row(headers, row, page_num, table_idx, row_idx)
                                        if row_text:
                                            table_elements.append({
                                                'text': clean_text_content(row_text),
                                                'page': page_num,
                                                'category': 'TableRow',
                                                'element_type': 'table_row',
                                                'table_index': table_idx,
                                                'row_index': row_idx
                                            })
                                            
                        except Exception as e:
                            logger.warning(f"Failed to process table {table_idx} on page {page_num}: {str(e)}")
                            
                except Exception as e:
                    logger.warning(f"Table extraction error on page {page_num}: {str(e)}")
                    
    except Exception as e:
        logger.error(f"pdfplumber failed for {pdf_path}: {str(e)}")
    
    return table_elements

def format_table(table: List[List], page_num: int, table_idx: int) -> str:
    """Format table for text output"""
    if not table or len(table) == 0:
        return ""
    
    formatted_lines = []
    formatted_lines.append(f"TABLE {table_idx + 1} (Page {page_num}):")
    formatted_lines.append("=" * 50)
    
    headers = table[0] if table else []
    
    for row_idx, row in enumerate(table):
        if row_idx == 0:
            formatted_lines.append("HEADERS: " + " | ".join(str(cell or "").strip() for cell in row))
            formatted_lines.append("-" * 50)
        else:
            row_data = []
            for col_idx, cell in enumerate(row):
                if col_idx < len(headers) and headers[col_idx]:
                    header = str(headers[col_idx]).strip()
                    value = str(cell or "").strip()
                    if header and value:
                        row_data.append(f"{header}: {value}")
            
            if row_data:
                formatted_lines.append(" | ".join(row_data))
    
    return "\n".join(formatted_lines)

def format_table_row(headers: List, row: List, page_num: int, table_idx: int, row_idx: int) -> str:
    """Format individual table row"""
    row_pairs = []
    for col_idx, cell in enumerate(row):
        if col_idx < len(headers) and headers[col_idx]:
            header = str(headers[col_idx]).strip()
            value = str(cell or "").strip()
            if header and value:
                row_pairs.append(f"{header}: {value}")
    
    if row_pairs:
        return f"Table Row (Page {page_num}, Table {table_idx + 1}): " + " | ".join(row_pairs)
    return ""

def create_chunks_with_unstructured(elements: List) -> List[Dict]:
    """Create chunks using Unstructured's chunk_by_title function"""
    try:
        # Use unstructured's chunking
        chunks = chunk_by_title(
            elements=elements,
            max_characters=2000,  # Maximum characters per chunk
            new_after_n_chars=1500,  # Start looking for break after this many chars
            combine_text_under_n_chars=100,  # Combine small elements
        )
        
        # Convert to our format
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            text = get_element_text(chunk)
            page = get_element_page(chunk)
            category = get_element_category(chunk)
            
            if text and text.strip():
                # Extract title from first line or use category
                lines = text.split('\n')
                title = lines[0][:100] + "..." if len(lines[0]) > 100 else lines[0]
                
                processed_chunks.append({
                    'chunk_id': i,
                    'title': title,
                    'text': clean_text_content(text),
                    'page': page,
                    'category': category,
                    'char_count': len(text),
                    'element_count': 1  # Each chunk is now a single combined element
                })
        
        return processed_chunks
        
    except Exception as e:
        logger.warning(f"Unstructured chunking failed: {str(e)}, falling back to custom chunking")
        # Fall back to custom chunking if unstructured fails
        return create_chunks_custom(elements)

def create_chunks_custom(elements: List[Dict]) -> List[Dict]:
    """Custom chunking logic as fallback"""
    chunks = []
    current_chunk = []
    current_title = ""
    current_page = 1
    chunk_id = 0
    
    for element in elements:
        category = element.get('category', '')
        text = element.get('text', '')
        page = element.get('page', 1)
        
        # Check if this is a new section/title
        if category in ['Title', 'Header'] or (len(text) < 100 and text.isupper()):
            # Save previous chunk if it exists
            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text.strip()) > 50:  # Only save substantial chunks
                    chunks.append({
                        'chunk_id': chunk_id,
                        'title': current_title,
                        'text': chunk_text.strip(),
                        'page': current_page,
                        'char_count': len(chunk_text),
                        'element_count': len(current_chunk)
                    })
                    chunk_id += 1
            
            # Start new chunk
            current_chunk = [text]
            current_title = text[:100] + "..." if len(text) > 100 else text
            current_page = page
        else:
            # Add to current chunk
            current_chunk.append(text)
            
            # Split large chunks
            current_text = '\n'.join(current_chunk)
            if len(current_text) > 2000:
                chunks.append({
                    'chunk_id': chunk_id,
                    'title': current_title,
                    'text': current_text.strip(),
                    'page': current_page,
                    'char_count': len(current_text),
                    'element_count': len(current_chunk)
                })
                chunk_id += 1
                current_chunk = []
                current_title = ""
    
    # Save final chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        if len(chunk_text.strip()) > 50:
            chunks.append({
                'chunk_id': chunk_id,
                'title': current_title,
                'text': chunk_text.strip(),
                'page': current_page,
                'char_count': len(chunk_text),
                'element_count': len(current_chunk)
            })
    
    return chunks

def save_parsed_text(elements: List[Dict], output_path: str, pdf_name: str):
    """Save parsed text to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"PARSED PDF: {pdf_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        current_page = 0
        for element in elements:
            page = element.get('page', 1)
            category = element.get('category', 'Unknown')
            text = element.get('text', '')
            
            if page != current_page:
                f.write(f"\n{'='*20} PAGE {page} {'='*20}\n\n")
                current_page = page
            
            f.write(f"[{category.upper()}]\n")
            f.write(text)
            f.write("\n\n" + "-" * 40 + "\n\n")

def save_chunks(chunks: List[Dict], output_path: str, pdf_name: str):
    """Save chunks to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"CHUNKS FROM: {pdf_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Chunks: {len(chunks)}\n")
        f.write("=" * 80 + "\n\n")
        
        for chunk in chunks:
            f.write(f"CHUNK #{chunk['chunk_id']}\n")
            f.write(f"Title: {chunk['title']}\n")
            f.write(f"Page: {chunk['page']}\n")
            f.write(f"Characters: {chunk['char_count']}\n")
            f.write(f"Elements: {chunk['element_count']}\n")
            f.write("-" * 50 + "\n")
            f.write(chunk['text'])
            f.write("\n\n" + "=" * 80 + "\n\n")

def parse_pdf(pdf_path: str):
    """Main function to parse PDF and generate output files"""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    pdf_name = pdf_path.stem
    output_dir = pdf_path.parent
    
    # Output file paths
    parsed_output = output_dir / f"{pdf_name}_parsed.txt"
    chunks_output = output_dir / f"{pdf_name}_chunks.txt"
    
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Extract text with Unstructured (get raw elements for chunking)
    elements = []
    for strategy in ["hi_res", "fast", "auto"]:
        try:
            logger.info(f"Trying Unstructured strategy: {strategy}")
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy=strategy,
                infer_table_structure=False,
                include_metadata=True,
            )
            if elements:
                logger.info(f"Success with strategy: {strategy}")
                break
        except Exception as e:
            logger.warning(f"Strategy '{strategy}' failed: {str(e)}")
            continue
    
    if not elements:
        logger.error("Failed to extract text with Unstructured")
        return
    
    # Process elements for parsed output
    text_elements = []
    for el in elements:
        category = get_element_category(el)
        if category.lower() in ("image", "figurecaption", "pagebreak"):
            continue
            
        text = get_element_text(el)
        page = get_element_page(el)
        
        if text and text.strip():
            text_elements.append({
                'text': clean_text_content(text),
                'page': page,
                'category': category,
                'element_type': 'text'
            })
    
    # Extract tables with pdfplumber
    table_elements = extract_tables_with_pdfplumber(str(pdf_path))
    
    # Combine all elements for parsed output
    all_elements = text_elements + table_elements
    all_elements.sort(key=lambda x: x.get('page', 1))
    
    logger.info(f"Extracted {len(text_elements)} text elements and {len(table_elements)} table elements")
    
    # Save parsed text
    save_parsed_text(all_elements, str(parsed_output), pdf_name)
    logger.info(f"Saved parsed text to: {parsed_output}")
    
    # Create chunks using Unstructured's chunking (on raw elements)
    chunks = create_chunks_with_unstructured(elements)
    save_chunks(chunks, str(chunks_output), pdf_name)
    logger.info(f"Saved {len(chunks)} chunks to: {chunks_output}")
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Parsed text saved to: {parsed_output}")
    print(f"✓ Chunks saved to: {chunks_output}")
    print(f"✓ Total elements processed: {len(all_elements)}")
    print(f"✓ Total chunks created: {len(chunks)}")

if __name__ == "__main__":
    # HERE ADD PATH - Replace with your PDF file path
    PDF_PATH = r"C:\4. Manuals\CD 01 Man B&W Diesel Engine\Book 1.pdf"
    
    try:
        parse_pdf(PDF_PATH)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        print(f"Error: {str(e)}")