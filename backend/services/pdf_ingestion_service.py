"""
PDF Document Ingestion Service for RAG Knowledge Base

Parses battery manuals, datasheets, and technical documents into ChromaDB
for intelligent Q&A via the chat agent.
"""

import os
import io
import base64
import json
from pathlib import Path
from typing import List, Dict
import hashlib
from datetime import datetime
from services.model_client import model_client
import asyncio
import logging

logger = logging.getLogger("pdf_ingestion_service")

class PDFIngestionService:
    def __init__(self):
        """Initialize PDF ingestion service"""
        self.upload_dir = Path(__file__).parent.parent / "uploads" / "manuals"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_bytes: bytes, filename: str) -> str:
        """
        Extract text from PDF using PyPDF2
        
        Args:
            pdf_bytes: PDF file content
            filename: Original filename
            
        Returns:
            str: Extracted text content
        """
        try:
            # Try PyPDF2 first
            import PyPDF2
            
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                except Exception as e:
                    logger.warning("Error extracting page %d: %s", page_num + 1, e)
                    continue
            
            return "\n\n".join(text_content)
            
        except ImportError:
            logger.warning("PyPDF2 not installed.")
            return None
        except Exception as e:
            logger.error("PyPDF2 extraction failed: %s", e)
            return None
    
    async def _extract_with_vision_model(self, pdf_bytes: bytes, filename: str) -> str:
        """
        Fallback: Use Vision model to read PDF (multimodal) via HTTP client
        """
        try:
            prompt = """
            Extract ALL text content from this technical document.
            Preserve headings, sections, tables, and technical specifications.
            Return the full text in a structured format.
            """
            
            if len(pdf_bytes) > 15_000_000:
                raise ValueError("PDF too large for model extraction")
            
            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            response_text = await model_client.generate_async(prompt, image_b64=pdf_b64, task="vision")
            
            return response_text
            
        except Exception as e:
            raise Exception(f"Failed to extract PDF content: {e}")
    
    def chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split document into overlapping chunks for better retrieval
        
        Args:
            text: Full document text
            chunk_size: Target characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or len(text) < chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find chunk end (prefer sentence boundaries)
            end = start + chunk_size
            
            if end < len(text):
                # Look for sentence end within next 100 chars
                sentence_end = text.find('. ', end, end + 100)
                if sentence_end != -1:
                    end = sentence_end + 1
            else:
                end = len(text)
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap ensuring no infinite loops
            start = max(end - overlap, start + 1)
        
        return chunks
    
    async def generate_document_metadata(self, text_sample: str, filename: str) -> Dict:
        """
        Use model to extract metadata from document
        
        Args:
            text_sample: First 2000 chars of document
            filename: Original filename
            
        Returns:
            dict with extracted metadata
        """
        try:
            prompt = f"""
            Analyze this technical document excerpt and extract metadata.
            
            Filename: {filename}
            Text Sample:
            {text_sample[:2000]}
            
            Return JSON with:
            {{
                "document_type": "Battery Manual|Datasheet|Safety Standard|Research Paper|Technical Guide",
                "title": "Extracted document title",
                "manufacturer": "Company name if applicable or 'Unknown'",
                "battery_chemistry": "NMC|LFP|LCO|NCA|Unknown",
                "topics": ["topic1", "topic2"],  // Max 5 key topics
                "summary": "One sentence summary"
            }}
            """
            
            result_text = await model_client.generate_async(prompt, task="text")
            
            # Extract JSON safely
            start = result_text.find("{")
            end = result_text.rfind("}")
            
            if start != -1 and end != -1:
                json_str = result_text[start:end+1]
                metadata = json.loads(json_str)
            else:
                raise ValueError("No JSON object found in metadata response")
                
            return metadata
            
        except Exception as e:
            logger.error("Metadata extraction failed: %s", e)
            return {
                "document_type": "Unknown",
                "title": filename,
                "manufacturer": "Unknown",
                "battery_chemistry": "Unknown",
                "topics": [],
                "summary": "No summary available"
            }
    
    async def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> Dict:
        """
        Complete PDF ingestion pipeline
        
        Args:
            pdf_bytes: PDF file content
            filename: Original filename
            
        Returns:
            dict with ingestion results
        """
        try:
            # 1. Extract text
            logger.info("Extracting text from %s...", filename)
            # Run CPU-bound PyPDF2 in a thread pool
            loop = asyncio.get_running_loop()
            try:
                full_text = await loop.run_in_executor(None, self.extract_text_from_pdf, pdf_bytes, filename)
            except Exception:
                 # Fallback to multimodal model extraction if PyPDF2 fails
                 full_text = None

            if not full_text:
                 # If sync extraction failed or returned nothing, try async vision model
                 full_text = await self._extract_with_vision_model(pdf_bytes, filename)
            
            if not full_text or len(full_text) < 50:
                raise Exception("Insufficient text content extracted")
            
            # 2. Generate metadata
            logger.info("Generating metadata...")
            metadata = await self.generate_document_metadata(full_text[:2000], filename)            
            # 3. Chunk document
            logger.info("Chunking document...")
            chunks = self.chunk_document(full_text, chunk_size=1000, overlap=200)
            
            # 4. Add to RAG service
            from services.rag_service import rag_service
            
            # Generate unique IDs
            file_hash = hashlib.md5(pdf_bytes).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            doc_ids = [f"{file_hash}_{timestamp}_chunk_{i}" for i in range(len(chunks))]
            
            # Metadata for each chunk
            chunk_metadatas = []
            for i in range(len(chunks)):
                chunk_meta = {
                    "title": metadata.get("title", filename),
                    "filename": filename,
                    "document_type": metadata.get("document_type", "Unknown"),
                    "manufacturer": metadata.get("manufacturer", "Unknown"),
                    "battery_chemistry": metadata.get("battery_chemistry", "Unknown"),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "upload_date": timestamp
                }
                chunk_metadatas.append(chunk_meta)
            
            # Add to ChromaDB
            logger.info("Adding %d chunks to vector database...", len(chunks))
            rag_service.add_documents(
                documents=chunks,
                metadatas=chunk_metadatas,
                ids=doc_ids
            )
            
            # Save original PDF
            safe_name = filename.replace("/", "_").replace("\\", "_")
            pdf_path = self.upload_dir / f"{file_hash}_{safe_name}"
            with open(pdf_path, 'wb') as f:
                f.write(pdf_bytes)
            
            return {
                "success": True,
                "filename": filename,
                "chunks_created": len(chunks),
                "total_characters": len(full_text),
                "metadata": metadata,
                "file_id": file_hash
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    def list_ingested_documents(self) -> List[Dict]:
        """
        List all ingested documents from ChromaDB
        
        Returns:
            List of document metadata
        """
        try:
            from services.rag_service import rag_service
            
            # Get all documents from collection
            results = rag_service.collection.get()
            
            # Group by filename and deduplicate
            docs_by_file = {}
            for i, metadata in enumerate(results['metadatas']):
                filename = metadata.get('filename', metadata.get('title', 'Unknown'))
                
                if filename not in docs_by_file:
                    docs_by_file[filename] = {
                        "filename": filename,
                        "title": metadata.get('title', filename),
                        "document_type": metadata.get('document_type', "Unknown"),
                        "manufacturer": metadata.get('manufacturer', "Unknown"),
                        "battery_chemistry": metadata.get('battery_chemistry', "Unknown"),
                        "chunks": 1,
                        "upload_date": metadata.get('upload_date', "Unknown")
                    }
                else:
                    docs_by_file[filename]['chunks'] += 1
            
            return list(docs_by_file.values())
            
        except Exception as e:
            logger.error("Error listing documents: %s", e)
            return []


# Singleton instance
pdf_ingestion_service = PDFIngestionService()