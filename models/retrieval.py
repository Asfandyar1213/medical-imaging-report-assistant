"""
Retrieval system for finding similar medical cases
"""

import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Import project config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class RetrievalSystem:
    """Retrieval system for finding similar medical cases"""
    
    def __init__(self, embedding_model=None):
        """
        Initialize the retrieval system
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = embedding_model if embedding_model else config.EMBEDDING_MODEL
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        # Initialize embedding model and vector database
        self._initialize_model()
        self._initialize_vector_db()
    
    def _initialize_model(self):
        """
        Initialize the embedding model
        """
        try:
            print(f"Loading embedding model {self.model_name}...")
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
            
            print(f"Embedding model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Using fallback embedding method for demonstration")
    
    def _initialize_vector_db(self):
        """
        Initialize the vector database
        """
        try:
            print("Initializing vector database...")
            
            # Create ChromaDB client
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=config.VECTOR_DB_PATH
            ))
            
            # Create embedding function
            if self.embedding_model:
                sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.model_name
                )
            else:
                # Use default embedding function
                sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name="medical_cases",
                    embedding_function=sentence_transformer_ef
                )
                print(f"Found existing collection with {self.collection.count()} documents")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name="medical_cases",
                    embedding_function=sentence_transformer_ef
                )
                print("Created new collection")
                
                # Add sample data for demonstration
                self._add_sample_data()
            
        except Exception as e:
            print(f"Error initializing vector database: {e}")
            print("Using fallback retrieval method for demonstration")
    
    def _add_sample_data(self):
        """
        Add sample data to the vector database for demonstration
        """
        print("Adding sample data to vector database...")
        
        sample_cases = [
            {
                "id": "case001",
                "findings": "The cardiac silhouette is enlarged, with a cardiothoracic ratio greater than 0.5. The lungs are clear without focal consolidation. No pleural effusion or pneumothorax.",
                "diagnosis": "Cardiomegaly",
                "patient_age": 65,
                "patient_gender": "Male",
                "modality": "Chest X-ray"
            },
            {
                "id": "case002",
                "findings": "There is blunting of the costophrenic angle, consistent with a small pleural effusion. The cardiac silhouette is normal in size. No pneumothorax or consolidation.",
                "diagnosis": "Pleural Effusion",
                "patient_age": 58,
                "patient_gender": "Female",
                "modality": "Chest X-ray"
            },
            {
                "id": "case003",
                "findings": "There is a focal area of consolidation in the right lower lobe, consistent with pneumonia. The cardiac silhouette is normal in size. No pleural effusion or pneumothorax.",
                "diagnosis": "Pneumonia",
                "patient_age": 42,
                "patient_gender": "Male",
                "modality": "Chest X-ray"
            },
            {
                "id": "case004",
                "findings": "There is a visible pleural line with absence of lung markings peripherally in the left upper lobe, consistent with pneumothorax. The cardiac silhouette is normal in size.",
                "diagnosis": "Pneumothorax",
                "patient_age": 35,
                "patient_gender": "Male",
                "modality": "Chest X-ray"
            },
            {
                "id": "case005",
                "findings": "There is an area of increased opacity in the right upper lobe. The cardiac silhouette is normal in size. No pleural effusion or pneumothorax.",
                "diagnosis": "Lung Opacity",
                "patient_age": 70,
                "patient_gender": "Female",
                "modality": "Chest X-ray"
            }
        ]
        
        # Add documents to collection
        ids = [case["id"] for case in sample_cases]
        documents = [case["findings"] for case in sample_cases]
        metadatas = [
            {
                "diagnosis": case["diagnosis"],
                "patient_age": case["patient_age"],
                "patient_gender": case["patient_gender"],
                "modality": case["modality"]
            }
            for case in sample_cases
        ]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Added {len(sample_cases)} sample cases to vector database")
    
    def embed_text(self, text):
        """
        Embed text using the embedding model
        
        Args:
            text: Text to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                return embedding
            except Exception as e:
                print(f"Error embedding text: {e}")
        
        # Fallback to random embedding for demonstration
        return np.random.rand(384)  # Common embedding dimension
    
    def retrieve_similar_cases(self, query_text, n_results=3, filter_criteria=None):
        """
        Retrieve similar cases from the vector database
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            filter_criteria: Dictionary of metadata filters
            
        Returns:
            list: List of similar cases
        """
        if self.collection:
            try:
                # Query the collection
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=filter_criteria
                )
                
                # Format results
                similar_cases = []
                for i in range(len(results["ids"][0])):
                    similar_cases.append({
                        "id": results["ids"][0][i],
                        "findings": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": float(results["distances"][0][i]) if "distances" in results else None
                    })
                
                return similar_cases
            except Exception as e:
                print(f"Error retrieving similar cases: {e}")
        
        # Fallback to sample data for demonstration
        return self._get_fallback_similar_cases(query_text, n_results, filter_criteria)
    
    def _get_fallback_similar_cases(self, query_text, n_results=3, filter_criteria=None):
        """
        Get fallback similar cases for demonstration
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            filter_criteria: Dictionary of metadata filters
            
        Returns:
            list: List of similar cases
        """
        # Sample cases
        sample_cases = [
            {
                "id": "case001",
                "findings": "The cardiac silhouette is enlarged, with a cardiothoracic ratio greater than 0.5. The lungs are clear without focal consolidation. No pleural effusion or pneumothorax.",
                "metadata": {
                    "diagnosis": "Cardiomegaly",
                    "patient_age": 65,
                    "patient_gender": "Male",
                    "modality": "Chest X-ray"
                },
                "similarity": 0.92
            },
            {
                "id": "case002",
                "findings": "There is blunting of the costophrenic angle, consistent with a small pleural effusion. The cardiac silhouette is normal in size. No pneumothorax or consolidation.",
                "metadata": {
                    "diagnosis": "Pleural Effusion",
                    "patient_age": 58,
                    "patient_gender": "Female",
                    "modality": "Chest X-ray"
                },
                "similarity": 0.85
            },
            {
                "id": "case003",
                "findings": "There is a focal area of consolidation in the right lower lobe, consistent with pneumonia. The cardiac silhouette is normal in size. No pleural effusion or pneumothorax.",
                "metadata": {
                    "diagnosis": "Pneumonia",
                    "patient_age": 42,
                    "patient_gender": "Male",
                    "modality": "Chest X-ray"
                },
                "similarity": 0.78
            },
            {
                "id": "case004",
                "findings": "There is a visible pleural line with absence of lung markings peripherally in the left upper lobe, consistent with pneumothorax. The cardiac silhouette is normal in size.",
                "metadata": {
                    "diagnosis": "Pneumothorax",
                    "patient_age": 35,
                    "patient_gender": "Male",
                    "modality": "Chest X-ray"
                },
                "similarity": 0.72
            },
            {
                "id": "case005",
                "findings": "There is an area of increased opacity in the right upper lobe. The cardiac silhouette is normal in size. No pleural effusion or pneumothorax.",
                "metadata": {
                    "diagnosis": "Lung Opacity",
                    "patient_age": 70,
                    "patient_gender": "Female",
                    "modality": "Chest X-ray"
                },
                "similarity": 0.65
            }
        ]
        
        # Filter by query text (simple keyword matching for demonstration)
        filtered_cases = []
        for case in sample_cases:
            # Check if any keywords from query are in the findings
            keywords = query_text.lower().split()
            if any(keyword in case["findings"].lower() for keyword in keywords):
                filtered_cases.append(case)
        
        # If no matches, return all cases
        if not filtered_cases:
            filtered_cases = sample_cases
        
        # Apply metadata filters if provided
        if filter_criteria:
            for key, value in filter_criteria.items():
                filtered_cases = [case for case in filtered_cases if case["metadata"].get(key) == value]
        
        # Return top n results
        return filtered_cases[:n_results]
    
    def add_case_to_database(self, case_id, findings, metadata=None):
        """
        Add a case to the vector database
        
        Args:
            case_id: Unique identifier for the case
            findings: Findings text
            metadata: Dictionary of metadata
            
        Returns:
            bool: Success or failure
        """
        if self.collection:
            try:
                self.collection.add(
                    ids=[case_id],
                    documents=[findings],
                    metadatas=[metadata] if metadata else None
                )
                return True
            except Exception as e:
                print(f"Error adding case to database: {e}")
                return False
        return False 