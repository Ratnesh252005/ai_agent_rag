import os
import uuid
import numpy as np
import streamlit as st
from datetime import datetime
from typing import List, Tuple, Dict, Any
from pinecone import Pinecone, ServerlessSpec
import time

class PineconeVectorStore:
    """Handles Pinecone vector database operations using the new Pinecone API"""

    def __init__(self, api_key: str, environment: str, index_name: str = "rag-documents"):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = None
        self.index = None

    def initialize_pinecone(self) -> bool:
        """Initialize Pinecone client"""
        try:
            self.pc = Pinecone(api_key=self.api_key)
            st.success("✅ Connected to Pinecone")
            return True
        except Exception as e:
            st.error(f"Error connecting to Pinecone: {str(e)}")
            return False

    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create or connect to Pinecone index using new SDK"""
        try:
            # Fix: Get list of existing indexes properly
            try:
                # Method 1: Try the new API way
                existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            except AttributeError:
                try:
                    # Method 2: Alternative new API way
                    index_list = self.pc.list_indexes()
                    if hasattr(index_list, 'names'):
                        existing_indexes = [idx.name for idx in index_list.names()]
                    else:
                        existing_indexes = [idx['name'] for idx in index_list]
                except:
                    # Method 3: Direct list approach
                    existing_indexes = []
                    index_response = self.pc.list_indexes()
                    if hasattr(index_response, '__iter__'):
                        for idx in index_response:
                            if hasattr(idx, 'name'):
                                existing_indexes.append(idx.name)
                            elif isinstance(idx, dict) and 'name' in idx:
                                existing_indexes.append(idx['name'])
                            elif isinstance(idx, str):
                                existing_indexes.append(idx)

            # Check if index exists
            if self.index_name not in existing_indexes:
                st.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create index with proper error handling
                try:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=dimension,
                        metric=metric,
                        spec=ServerlessSpec(
                            cloud='aws', 
                            region=self.environment if self.environment.startswith('us-') else 'us-east-1'
                        )
                    )
                    st.success(f"✅ Created index '{self.index_name}' with dimension {dimension}")
                    
                    # Wait for index to be ready
                    st.info("⏳ Waiting for index to be ready...")
                    time.sleep(10)  # Give it time to initialize
                    
                except Exception as create_error:
                    st.error(f"Error creating index: {str(create_error)}")
                    # Try with different region if first attempt fails
                    try:
                        st.info("Retrying with us-east-1 region...")
                        self.pc.create_index(
                            name=self.index_name,
                            dimension=dimension,
                            metric=metric,
                            spec=ServerlessSpec(cloud='aws', region='us-east-1')
                        )
                        st.success(f"✅ Created index '{self.index_name}' with dimension {dimension}")
                        time.sleep(10)
                    except Exception as retry_error:
                        st.error(f"Failed to create index even with retry: {str(retry_error)}")
                        return False
            else:
                st.info(f"✅ Using existing index: {self.index_name}")

            # Connect to index with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.index = self.pc.Index(self.index_name)
                    # Test the connection
                    self.index.describe_index_stats()
                    st.success(f"✅ Successfully connected to index: {self.index_name}")
                    return True
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                        time.sleep(5)
                    else:
                        st.error(f"Failed to connect to index after {max_retries} attempts: {str(e)}")
                        return False

        except Exception as e:
            st.error(f"Error in create_index: {str(e)}")
            return False

    def upsert_embeddings(
        self,
        embedded_chunks: List[Tuple[np.ndarray, str, int]],
        document_id: str = None,
        document_name: str = None
    ) -> bool:
        """Upload embeddings to Pinecone"""
        if not self.index:
            st.error("Pinecone index not initialized")
            return False

        if not embedded_chunks:
            st.warning("No embeddings to upload")
            return False

        try:
            if not document_id:
                document_id = str(uuid.uuid4())

            st.info("📤 Uploading embeddings to Pinecone...")
            vectors = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (embedding, text, chunk_num) in enumerate(embedded_chunks):
                vector_id = f"{document_id}_chunk_{chunk_num}"
                
                # Ensure embedding is in the right format
                if isinstance(embedding, np.ndarray):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = list(embedding)
                
                metadata = {
                    "text": text[:1000],  # Pinecone has metadata size limits
                    "chunk_number": chunk_num,
                    "document_id": document_id,
                    "document_name": document_name or "Untitled Document",
                    "timestamp": datetime.now().isoformat(),
                    "text_length": len(text)
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding_list,
                    "metadata": metadata
                })

                progress_bar.progress((i + 1) / len(embedded_chunks))
                status_text.text(f"Preparing vector {i + 1}/{len(embedded_chunks)}")

            # Upsert in batches with error handling
            batch_size = 100
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    self.index.upsert(vectors=batch)
                    progress_bar.progress(batch_num / total_batches)
                    status_text.text(f"Uploading batch {batch_num}/{total_batches}")
                except Exception as batch_error:
                    st.error(f"Error uploading batch {batch_num}: {str(batch_error)}")
                    # Try individual vectors in this batch
                    for vector in batch:
                        try:
                            self.index.upsert(vectors=[vector])
                        except Exception as single_error:
                            st.warning(f"Skipped vector {vector['id']}: {str(single_error)}")

            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Successfully uploaded {len(vectors)} embeddings to Pinecone")

            # Track documents in session state
            if 'documents' not in st.session_state:
                st.session_state.documents = []
            
            # Remove existing document with same ID if exists
            st.session_state.documents = [d for d in st.session_state.documents if d.get('id') != document_id]
            
            # Add new document
            st.session_state.documents.append({
                'id': document_id,
                'name': document_name or 'Untitled Document',
                'vector_count': len(vectors),
                'created_at': datetime.now().isoformat()
            })
            st.session_state.current_document_id = document_id

            return True
            
        except Exception as e:
            st.error(f"Error uploading embeddings: {str(e)}")
            return False

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_id: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone"""
        if not self.index:
            st.error("Pinecone index not initialized")
            return []

        try:
            # Ensure query embedding is in the right format
            if isinstance(query_embedding, np.ndarray):
                query_vector = query_embedding.tolist()
            else:
                query_vector = list(query_embedding)
            
            # Prepare filter
            filter_dict = {"document_id": {"$eq": document_id}} if document_id else None
            
            # Query with error handling
            try:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
            except Exception as query_error:
                st.warning(f"Query with filter failed: {str(query_error)}")
                # Retry without filter
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )

            similar_chunks = []
            for match in results.matches:
                similar_chunks.append({
                    "id": match.id,
                    "score": float(match.score),
                    "text": match.metadata.get("text", ""),
                    "chunk_number": match.metadata.get("chunk_number", 0),
                    "document_id": match.metadata.get("document_id", ""),
                    "document_name": match.metadata.get("document_name", "Unknown"),
                    "timestamp": match.metadata.get("timestamp", "")
                })
                
            return similar_chunks
            
        except Exception as e:
            st.error(f"Error searching vectors: {str(e)}")
            return []

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index"""
        if not self.index:
            return {"error": "Index not initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if hasattr(stats, 'namespaces') and stats.namespaces else {}
            }
        except Exception as e:
            st.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}

    def delete_document(self, document_id: str) -> bool:
        """Delete all vectors for a specific document"""
        if not self.index:
            st.error("Pinecone index not initialized")
            return False
            
        try:
            # Delete using filter
            self.index.delete(filter={"document_id": {"$eq": document_id}})
            st.success(f"✅ Deleted document {document_id} from vector store")
            
            # Remove from session state
            if 'documents' in st.session_state:
                st.session_state.documents = [
                    d for d in st.session_state.documents 
                    if d.get('id') != document_id
                ]
            
            return True
            
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False

    def delete_all_vectors(self) -> bool:
        """Delete all vectors from the index"""
        if not self.index:
            st.error("Pinecone index not initialized")
            return False
            
        try:
            self.index.delete(delete_all=True)
            st.success("✅ Deleted all vectors from the index")
            
            # Clear session state
            if 'documents' in st.session_state:
                st.session_state.documents = []
            
            return True
            
        except Exception as e:
            st.error(f"Error deleting all vectors: {str(e)}")
            return False

    def test_connection(self) -> bool:
        """Test the Pinecone connection"""
        if not self.pc:
            return False
            
        try:
            # Try to list indexes
            self.pc.list_indexes()
            return True
        except Exception as e:
            st.error(f"Connection test failed: {str(e)}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the vector store"""
        health_status = {
            "pinecone_client": False,
            "index_exists": False,
            "index_connected": False,
            "can_query": False,
            "stats_available": False
        }
        
        try:
            # Test Pinecone client
            if self.pc:
                health_status["pinecone_client"] = True
                
                # Test if index exists
                try:
                    existing_indexes = [idx.name for idx in self.pc.list_indexes()]
                    if self.index_name in existing_indexes:
                        health_status["index_exists"] = True
                except:
                    pass
                
                # Test index connection
                if self.index:
                    health_status["index_connected"] = True
                    
                    # Test query capability
                    try:
                        test_vector = [0.1] * 384  # Assuming 384-dimensional vectors
                        self.index.query(vector=test_vector, top_k=1)
                        health_status["can_query"] = True
                    except:
                        pass
                    
                    # Test stats
                    try:
                        self.index.describe_index_stats()
                        health_status["stats_available"] = True
                    except:
                        pass
        
        except Exception as e:
            health_status["error"] = str(e)
        
        return health_status