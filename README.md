# GraphRAG: Graph-Based RAG System with Neo4j

## Overview
GraphRAG is a document question-answering system that combines Retrieval-Augmented Generation (RAG) with graph database capabilities using Neo4j. The system processes documents, stores them as connected chunks, and provides a chat interface for querying document knowledge.

## Key Features
- Document Processing: Supports PDF, DOCX, and TXT files
- Local Embeddings: Uses SentenceTransformer for local embedding generation
- Graph-Based Storage: Utilizes Neo4j for storing document chunks and relationships
- Context-Aware Retrieval: Implements sliding window context for better answers
- Interactive Interface: Gradio-based UI for document upload and querying
- Extensible Architecture: Modular design with clear separation of concerns



### Improved Similarity Search:

- Replaces manual dot product calculations with Neo4j's native vector search
- Uses the vector index for faster and more efficient similarity calculations
- Returns properly formatted results matching your existing structure


### Optimized Document Processing:

- Adds batch processing for embeddings to manage memory better
- Ensures embeddings are stored in the correct format for the vector index
- Maintains compatibility with your existing code structure

### Local Embeddings

- all-MiniLM-L6-v2: General-purpose model with 384 dimensions.