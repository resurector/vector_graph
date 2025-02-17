# GraphRAG

GraphRAG is a powerful Retrieval-Augmented Generation (RAG) system that uses Neo4j graph database for efficient document storage, retrieval, and querying. It combines the power of graph databases, vector embeddings, and large language models to provide intelligent responses based on your document corpus.

## Features

- 📄 Multi-format document support (PDF, DOCX, TXT)
- 🔍 Intelligent chunking with configurable sizes and overlap
- 📊 Graph-based document representation
- 🧮 Local embeddings using Sentence Transformers
- 💾 Semantic caching for faster responses
- 🔗 Context-aware retrieval with document linkages
- 🚀 Gradio-based interactive interface
- ⚡ Asynchronous processing

## Architecture

GraphRAG uses a multi-component architecture:

1. **Document Processing**:
   - Extracts text from multiple document formats
   - Chunks text with configurable size and overlap
   - Generates embeddings using Sentence Transformers

2. **Storage Layer**:
   - Neo4j graph database for document and chunk storage
   - Vector indexing for similarity search
   - Semantic caching for query responses

3. **Retrieval Engine**:
   - Hybrid search combining vector similarity and text relevance
   - Context-aware retrieval with surrounding chunks
   - Enhanced ranking system

4. **Interface**:
   - Gradio-based web interface
   - Chat interface for queries
   - Document upload and processing capabilities

## Prerequisites

- Python 3.8+
- Neo4j 4.4+ with Graph Data Science Library
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/resurector/vector_graph.git
cd vector_graph
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
```

## Usage

1. Start the application:
```bash
python vector_2.py
```

2. Access the web interface at `http://localhost:7860`

3. Upload documents through the "Process Document" tab

4. Ask questions about your documents in the "Chat" tab

## Configuration

Key configuration options in `Config` class:

```python
class Config:
    def __init__(self):
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
```

## API Reference

### Document Processing

```python
async def process_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> Dict
```
Processes a document and stores it in the graph database.

### Query Processing

```python
async def query_knowledge(self, user_query: str) -> Dict[str, Any]
```
Queries the knowledge base and returns relevant information.

## Performance Optimization

1. **Chunk Size Optimization**:
   - Default: 500 characters
   - Adjust based on document type and content density
   - Balance between context preservation and retrieval precision

2. **Vector Search**:
   - Uses Neo4j's vector index for similarity search
   - Configurable similarity threshold
   - Hybrid ranking combining vector and text similarity

3. **Caching**:
   - Semantic caching for similar queries
   - Configurable cache duration
   - Cache invalidation on document updates

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Neo4j for graph database capabilities
- OpenAI for language model integration
- Sentence Transformers for local embeddings
- Gradio for the web interface

## Future Improvements

- [ ] Add support for more document formats
- [ ] Implement advanced caching strategies
- [ ] Add document update/delete capabilities
- [ ] Enhance query relevance ranking
- [ ] Add batch processing for large document sets
- [ ] Implement user authentication
- [ ] Add API endpoints for programmatic access

## Troubleshooting

### Common Issues

1. **Neo4j Connection**:
   - Ensure Neo4j is running and accessible
   - Check credentials in `.env`
   - Verify GDS library installation

2. **Document Processing**:
   - Check file permissions
   - Verify supported file formats
   - Monitor chunking parameters

3. **Query Performance**:
   - Optimize chunk sizes
   - Adjust vector similarity thresholds
   - Check Neo4j index usage


