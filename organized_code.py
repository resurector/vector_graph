import os
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF
import gradio as gr
from docx import Document as DocxDocument
from neo4j import GraphDatabase

# Sentence-Transformers for local embeddings
from sentence_transformers import SentenceTransformer

# LangChain/OpenAI-related
from langchain_openai import ChatOpenAI
from langchain.vectorstores import VectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# -------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format for readability
    handlers=[
        logging.FileHandler('graphrag.log'),  # Log to file
        logging.StreamHandler()               # Log to console
    ]
)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
class Config:
    """Reads and validates environment variables / configuration."""
    def __init__(self):
        load_dotenv()
        
        # Neo4j
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', '')
        
        # Provider configuration
        self.provider_type = os.getenv('PROVIDER_TYPE', 'openai')  # 'openai' or 'azure'
        
        # OpenAI settings
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
        self._validate()

    def _validate(self):
        # Basic checks
        if not self.neo4j_uri or not self.neo4j_username:
            raise ValueError("Missing Neo4j connection details.")
        if self.provider_type == 'openai' and not self.openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI usage.")

# -------------------------------------------------------------------------
# Neo4j Connection / Helper
# -------------------------------------------------------------------------
class Neo4jHelper:
    """Helper class to handle Neo4j connections & queries."""
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()
    
    def run_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Run a Cypher query and return the data as a list of dictionaries."""
        with self.driver.session() as session:
            return session.run(query, parameters or {}).data()

# -------------------------------------------------------------------------
# Local Embeddings (using SentenceTransformers)
# -------------------------------------------------------------------------
class LocalEmbeddings:
    """
    A wrapper around SentenceTransformers to provide local embeddings.
    Replaces the OpenAIEmbeddings for offline or cost-effective usage.
    """
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):  # Updated model name
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query string."""
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents (strings)."""
        return self.model.encode(texts).tolist()

# -------------------------------------------------------------------------
# Placeholder for entity parsing
# -------------------------------------------------------------------------
def parse_entities(llm_response: str) -> List[str]:
    """
    Placeholder function for parsing entity names from an LLM response.
    Modify this logic based on how you expect the LLM to return entities.
    """
    # Example: If your LLM returns a comma-separated list of entities,
    # you could do something like:
    #
    # return [ent.strip() for ent in llm_response.split(",") if ent.strip()]
    #
    # For now, we'll just return an empty list.
    return []

# -------------------------------------------------------------------------
# GraphRAG Processor
# -------------------------------------------------------------------------
class GraphRAGProcessor:
    """Core class handling document ingestion, indexing, and query to Neo4j."""
    def __init__(self):
        self.config = Config()
        
        # Set environment variables for OpenAI
        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
        
        # Create or open Neo4j connection
        self.neo4j = Neo4jHelper(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_username,
            password=self.config.neo4j_password
        )
        
        # Create LLM & embeddings
        self.chat_model = ChatOpenAI(
            model_name=self.config.openai_model,
            temperature=0.2,
            api_key=self.config.openai_api_key  # Explicitly pass API key
        )
        
        
        # Using local embeddings instead:
        self.embeddings = LocalEmbeddings(model_name='all-mpnet-base-v2')

        # Ensure database constraints and indexes are in place
        self._setup_db()

    def _setup_db(self):
        """Setup necessary constraints and indexes in Neo4j (e.g., vector index)."""
        try:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.index IS NOT NULL"
            ]
            for constraint in constraints:
                self.neo4j.run_query(constraint)
            
            # Create vector index (note dimension=384 for the all-MiniLM-L6-v2 embeddings)
            vector_index_query = """
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk)
            ON (c.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
            self.neo4j.run_query(vector_index_query)
        except Exception as e:
            logging.error(f"Error setting up database: {e}")
            raise

    # ---------------------------------------------------------------------
    # Text Extraction
    # ---------------------------------------------------------------------
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF, DOCX, or TXT files."""
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            text = []
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text.append(page.get_text())
            return "\n".join(text)
        elif ext == '.docx':
            doc = DocxDocument(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")

    # ---------------------------------------------------------------------
    # Semantic Relationships (Optional / Example)
    # ---------------------------------------------------------------------
    def create_semantic_relationships(self, doc_id: str):
        """
        Example function to create relationships between chunks if they are 
        semantically similar within the same document.
        """
        chunks = self.neo4j.run_query("""
            MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
            RETURN c.id as chunk_id, c.text as text, c.embedding as embedding
        """, {"doc_id": doc_id})
        
        # If you want to compare chunks pairwise for similarity
        for chunk1 in chunks:
            for chunk2 in chunks:
                if chunk1['chunk_id'] != chunk2['chunk_id']:
                    similarity = self.calculate_similarity(
                        chunk1['embedding'], 
                        chunk2['embedding']
                    )
                    if similarity > 0.8:  # Threshold for similarity
                        self.neo4j.run_query("""
                            MATCH (c1:Chunk {id: $chunk1_id})
                            MATCH (c2:Chunk {id: $chunk2_id})
                            MERGE (c1)-[:SEMANTICALLY_RELATED]->(c2)
                        """, {
                            "chunk1_id": chunk1['chunk_id'],
                            "chunk2_id": chunk2['chunk_id']
                        })

    def calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Compute cosine similarity (or another similarity measure) between two embeddings.
        """
        # Example: You could implement your own or use a library function
        import numpy as np
        v1 = np.array(emb1)
        v2 = np.array(emb2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # ---------------------------------------------------------------------
    # Entity Extraction & Linking (Optional / Example)
    # ---------------------------------------------------------------------
    def extract_and_link_entities(self, chunk_text: str, chunk_id: str):
        """
        Example function to use the LLM for entity extraction and store them in Neo4j.
        """
        prompt = f"Extract key entities (people, organizations, concepts) from this text: {chunk_text}"
        response = self.chat_model.predict(prompt)
        entities = parse_entities(response)  # parse_entities is a placeholder

        for entity in entities:
            self.neo4j.run_query("""
                MERGE (e:Entity {name: $entity})
                WITH e
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
            """, {
                "entity": entity,
                "chunk_id": chunk_id
            })

    # ---------------------------------------------------------------------
    # Document Processing
    # ---------------------------------------------------------------------
    async def process_document(
        self, 
        file_path: str, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100
    ) -> Dict[str, Any]:
        """
        Process a document: extract text, split into chunks, create nodes/edges in Neo4j, 
        and store embeddings.
        """
        doc_id = ""
        try:
            # Extract and chunk text
            text = self.extract_text_from_file(file_path)
            if not text.strip():
                return {"status": "error", "message": "Empty file or text extraction failed."}
    
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(text)
            document_name = os.path.basename(file_path)
            
            unique_id = uuid.uuid4().hex
            doc_id = f"doc-{document_name}-{unique_id}"
    
            # Create document node
            self.neo4j.run_query(
                """
                CREATE (d:Document {
                    id: $doc_id,
                    file_name: $file_name,
                    upload_date: datetime(),
                    chunk_count: $chunk_count,
                    file_path: $file_path
                })
                """,
                {
                    "doc_id": doc_id,
                    "file_name": document_name,
                    "chunk_count": len(chunks),
                    "file_path": file_path
                }
            )
    
            # Create chunk nodes
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}-chunk-{idx}"
                embedding = self.embeddings.embed_query(chunk_text)
                
                # Create the chunk node
                self.neo4j.run_query(
                    """
                    CREATE (c:Chunk {
                        id: $chunk_id,
                        text: $chunk_text,
                        embedding: $embedding,
                        index: $idx,
                        position_start: $pos_start,
                        position_end: $pos_end
                    })
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:HAS_CHUNK {sequence: $idx}]->(c)
                    """,
                    {
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                        "embedding": embedding,
                        "idx": idx,
                        "doc_id": doc_id,
                        "pos_start": idx * (chunk_size - chunk_overlap),
                        "pos_end": (idx + 1) * chunk_size - (idx * chunk_overlap)
                    }
                )
    
                # Link chunk to the previous chunk
                if idx > 0:
                    self.neo4j.run_query(
                        """
                        MATCH (prev:Chunk {id: $prev_id})
                        MATCH (curr:Chunk {id: $curr_id})
                        MERGE (prev)-[:NEXT]->(curr)
                        """,
                        {
                            "prev_id": f"{doc_id}-chunk-{idx-1}",
                            "curr_id": chunk_id
                        }
                    )
    
            # Verify chunk counts
            verification = self.neo4j.run_query(
                """
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WITH d, count(c) as chunk_count
                RETURN d.chunk_count = chunk_count as is_valid
                """,
                {"doc_id": doc_id}
            )
    
            if not verification or not verification[0].get('is_valid', False):
                logging.warning(f"Document {doc_id} structure verification failed.")
    
            return {
                "status": "success",
                "file": document_name,
                "chunks_stored": len(chunks),
                "doc_id": doc_id
            }
    
        except Exception as e:
            logging.error(f"Error processing document: {e}")
            # Cleanup on failure
            if doc_id:
                try:
                    self.neo4j.run_query(
                        """
                        MATCH (d:Document {id: $doc_id})
                        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                        DETACH DELETE d, c
                        """,
                        {"doc_id": doc_id}
                    )
                except:
                    pass
            return {"status": "error", "message": str(e)}

    # ---------------------------------------------------------------------
    # Similarity Search
    # ---------------------------------------------------------------------
    def similarity_search(self, query: str, limit: int = 5, context_window: int = 1) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search on chunk embeddings and retrieve 
        a context window of surrounding chunks within the same document.
        """
        query_embedding = self.embeddings.embed_query(query)
        
        traversal_query = """
        // Initial vector similarity search
        CALL db.index.vector.queryNodes(
            'chunk_embeddings',
            $k,
            $query_embedding
        ) YIELD node, score
        
        // Get the matching chunk with its document
        MATCH (doc:Document)-[:HAS_CHUNK]->(node)
        
        // Get surrounding context chunks through document
        MATCH (doc)-[:HAS_CHUNK]->(context:Chunk)
        WHERE context.index >= node.index - $window 
        AND context.index <= node.index + $window
        
        // Aggregate results
        WITH 
            node as main_chunk,
            doc.file_name as source_doc,
            score as similarity_score,
            collect(DISTINCT {
                text: context.text,
                index: context.index,
                relative_position: context.index - node.index
            }) as context_chunks
        
        // Order by score and get specified limit
        ORDER BY similarity_score DESC
        LIMIT $limit
        
        RETURN {
            chunk_id: main_chunk.id,
            chunk_text: main_chunk.text,
            chunk_index: main_chunk.index,
            similarity_score: similarity_score,
            source_document: source_doc,
            context_chunks: context_chunks
        } as result
        """
        
        try:
            results = self.neo4j.run_query(
                traversal_query,
                {
                    "k": limit * 2,  # Get more initial results for filtering
                    "limit": limit,
                    "window": context_window,
                    "query_embedding": query_embedding
                }
            )
            
            # Process and format results
            formatted_results = []
            for item in results:
                result = item['result']
                
                # Sort context chunks by their position (before or after)
                context = sorted(
                    result['context_chunks'], 
                    key=lambda x: x['relative_position']
                )
                
                # Build a structured result
                formatted_result = {
                    "chunk_id": result['chunk_id'],
                    "chunk_text": result['chunk_text'],
                    "score": result['similarity_score'],
                    "source": result['source_document'],
                    "index": result['chunk_index'],
                    "context": {
                        "before": [
                            c['text'] for c in context if c['relative_position'] < 0
                        ],
                        "after": [
                            c['text'] for c in context if c['relative_position'] > 0
                        ]
                    }
                }
                
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error in traversal search: {e}")
            # Fallback to basic search
            return self._basic_similarity_search(query, limit)

    def _basic_similarity_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fallback basic similarity search if advanced context search fails."""
        query_embedding = self.embeddings.embed_query(query)
        
        basic_query = """
        CALL db.index.vector.queryNodes(
            'chunk_embeddings',
            $k,
            $query_embedding
        ) YIELD node, score
        MATCH (doc:Document)-[:HAS_CHUNK]->(node)
        RETURN 
            node.id AS chunk_id,
            node.text AS chunk_text,
            score,
            doc.file_name AS source
        LIMIT $limit
        """
        
        try:
            results = self.neo4j.run_query(
                basic_query,
                {
                    "k": limit,
                    "limit": limit,
                    "query_embedding": query_embedding
                }
            )
            
            return [
                {
                    "chunk_id": item["chunk_id"],
                    "chunk_text": item["chunk_text"],
                    "score": item["score"],
                    "source": item["source"]
                } 
                for item in results
            ]
        except Exception as e:
            logging.error(f"Error in basic search: {e}")
            return []

    # ---------------------------------------------------------------------
    # Graph Verification
    # ---------------------------------------------------------------------
    def verify_graph_structure(self) -> Dict[str, Any]:
        """Check whether stored document/chunk counts match expected counts."""
        verification_query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d,
             count(c) as actual_chunks,
             d.chunk_count as expected_chunks,
             count(c.index) as chunks_with_index,
             count(c.embedding) as chunks_with_embedding
        RETURN
            count(d) as document_count,
            sum(CASE WHEN actual_chunks = d.chunk_count THEN 1 ELSE 0 END) as valid_documents,
            sum(CASE WHEN actual_chunks != d.chunk_count THEN 1 ELSE 0 END) as invalid_documents,
            sum(CASE WHEN chunks_with_index < actual_chunks THEN 1 ELSE 0 END) as documents_missing_indexes,
            sum(CASE WHEN chunks_with_embedding < actual_chunks THEN 1 ELSE 0 END) as documents_missing_embeddings
        """
        
        results = self.neo4j.run_query(verification_query)
        return results[0] if results else {}

    # ---------------------------------------------------------------------
    # Querying (RAG Approach)
    # ---------------------------------------------------------------------
    async def query_knowledge(self, user_query: str) -> Dict[str, Any]:
        """
        Perform a retrieval-augmented generation (RAG) query. 
        1) Retrieve relevant chunks using vector similarity search 
        2) Provide that context to the LLM 
        3) Return the LLM's answer plus the raw search results.
        """
        try:
            logging.info(f"\n{'='*80}\nUser Query: {user_query}\n{'='*80}")
            
            # 1. Retrieve chunks
            search_results = self.similarity_search(query=user_query, limit=5, context_window=2)
            logging.info(f"\nFound {len(search_results)} relevant chunks:")
            
            for i, result in enumerate(search_results, 1):
                logging.info(f"\nChunk {i}:")
                logging.info(f"Source: {result['source']}")
                logging.info(f"Score: {result['score']}")
                logging.info(f"Text: {result['chunk_text']}")
                if result['context']['before']:
                    logging.info("Context Before:")
                    for ctx in result['context']['before']:
                        logging.info(f"- {ctx}")
                if result['context']['after']:
                    logging.info("Context After:")
                    for ctx in result['context']['after']:
                        logging.info(f"- {ctx}")

            if not search_results:
                return {"answer": "No relevant information found in the database.", "search_results": []}
            
            # 2. Prepare context for the LLM
            context_sections = []
            for result in search_results:
                section = (
                    f"\nSource: {result['source']}\n"
                    f"Relevance Score: {round(result['score'], 2)}\n"
                    f"Content: {result['chunk_text']}\n"
                )
                if result['context']['before']:
                    section += "\nPrevious Context:\n" + "\n".join(result['context']['before'])
                if result['context']['after']:
                    section += "\nFollowing Context:\n" + "\n".join(result['context']['after'])
                
                context_sections.append(section)
            
            context_text = "\n---\n".join(context_sections)
            
            system_prompt = (
                "You are a helpful assistant analyzing document content.\n"
                "Please use the provided context to answer the question accurately.\n"
                "If the context doesn't fully answer the question, acknowledge the limitations.\n\n"
                f"Context:\n{context_text}"
            )
            
            logging.info(f"\n{'='*80}\nPrompt being sent to OpenAI:\n{'='*80}")
            logging.info(f"System Prompt:\n{system_prompt}")
            logging.info(f"User Query: {user_query}")

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            # 3. Get the LLM's answer
            response = await self.chat_model.agenerate([[m for m in messages]])
            answer_text = response.generations[0][0].text.strip()
            
            return {
                "answer": answer_text,
                "search_results": search_results
            }
        except Exception as e:
            logging.error(f"Query error: {e}")
            return {"answer": f"Error: {str(e)}", "search_results": []}

# -------------------------------------------------------------------------
# Gradio Chat Interface
# -------------------------------------------------------------------------
class ChatInterface:
    """A simple Gradio interface for the GraphRAGProcessor."""
    def __init__(self, processor: GraphRAGProcessor):
        self.processor = processor
        self.interface = self._setup_gradio()

    def _setup_gradio(self) -> gr.Blocks:
        with gr.Blocks(title="GraphRAG", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# GraphRAG: Ask Your Documents in Neo4j")

            with gr.Tab("Chat"):
                chat_history_box = gr.Textbox(
                    label="Conversation History",
                    lines=15,
                    interactive=False
                )
                user_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Type a question...",
                    lines=2
                )
                send_button = gr.Button("Send")

                current_answer = gr.Textbox(
                    label="Answer",
                    interactive=False
                )
                search_result_df = gr.DataFrame(
                    headers=["Chunk ID", "Similarity Score", "Preview"],
                    label="Search Results",
                    interactive=False
                )

                async def chat_handler(question, chat_history):
                    if not question.strip():
                        return chat_history, "", []
                    
                    history_text = f"{chat_history}\n\nUser: {question}"
                    
                    query_res = await self.processor.query_knowledge(question)
                    answer = query_res.get("answer", "")
                    top_chunks = query_res.get("search_results", [])

                    table_data = [
                        [
                            c["chunk_id"],
                            round(c["score"], 3),
                            (c["chunk_text"][:70] + "...") if len(c["chunk_text"]) > 70 else c["chunk_text"]
                        ]
                        for c in top_chunks
                    ]
                    
                    updated_history = history_text + f"\nAssistant: {answer}"
                    return updated_history, answer, table_data

                send_button.click(
                    fn=chat_handler,
                    inputs=[user_input, chat_history_box],
                    outputs=[chat_history_box, current_answer, search_result_df]
                )

            with gr.Tab("Process Document"):
                file_input = gr.File(label="Upload a PDF, DOCX, or TXT file")
                chunk_size_slider = gr.Slider(100, 2000, step=100, value=500, label="Chunk Size")
                overlap_slider = gr.Slider(0, 500, step=50, value=100, label="Chunk Overlap")
                status_box = gr.Textbox(label="Status", interactive=False)
                
                process_button = gr.Button("Process File")

                async def process_handler(file, chunk_size, overlap):
                    if not file:
                        return "No file uploaded."
                    result = await self.processor.process_document(
                        file.name,
                        chunk_size=chunk_size,
                        chunk_overlap=overlap
                    )
                    if result["status"] == "success":
                        return f"Processed '{result['file']}' with {result['chunks_stored']} chunks stored."
                    else:
                        return f"Error: {result['message']}"

                process_button.click(
                    fn=process_handler,
                    inputs=[file_input, chunk_size_slider, overlap_slider],
                    outputs=[status_box]
                )

        return demo

    def launch(self):
        """Launch the Gradio app."""
        # Set share=True to create a public link (optional).
        self.interface.launch(server_port=7860, share=True, inbrowser=True)

# -------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------
def main():
    """Create and launch the Gradio interface tied to a GraphRAGProcessor."""
    processor = GraphRAGProcessor()
    chat_app = ChatInterface(processor)
    chat_app.launch()
    logging.info("GraphRAG interface launched.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
