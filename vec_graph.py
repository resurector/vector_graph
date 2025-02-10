import os
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sentence_transformers import SentenceTransformer #local embeddings

import fitz  # PyMuPDF for PDF
import gradio as gr
from docx import Document as DocxDocument
from dotenv import load_dotenv

# OpenAI-specific imports from langchain-openai
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Other LangChain imports
from langchain.vectorstores import VectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from neo4j import GraphDatabase


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
class Config:
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
        self.openai_embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
        
        self._validate()

    def _validate(self):
        # Basic checks
        if not self.neo4j_uri or not self.neo4j_username:
            raise ValueError("Missing Neo4j connection details")
        if self.provider_type == 'openai' and not self.openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI usage")

# -------------------------------------------------------------------------
# Neo4j Connection / Helper
# -------------------------------------------------------------------------
class Neo4jHelper:
    """A small helper class to handle Neo4j connections & queries."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def run_query(self, query: str, parameters: Optional[Dict] = None):
        with self.driver.session() as session:
            return session.run(query, parameters or {}).data()


# -------------------------------------------------------------------------
# LocalEmbeddings
# -------------------------------------------------------------------------

class LocalEmbeddings:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the local embedding model.
        :param model_name: Name of the pre-trained model from Sentence-Transformers.
        """
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text query.
        :param text: Input text to embed.
        :return: List of floats representing the embedding.
        """
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        :param texts: List of input texts to embed.
        :return: List of embeddings (each embedding is a list of floats).
        """
        return self.model.encode(texts).tolist()            

# -------------------------------------------------------------------------
# GraphRAG Processor
# -------------------------------------------------------------------------
class GraphRAGProcessor:
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
        # self.embeddings = OpenAIEmbeddings(
        # model=self.config.openai_embedding_model,
        # api_key=self.config.openai_api_key  # Explicitly pass API key
        # )
        
        # Replace OpenAIEmbeddings with LocalEmbeddings
        self.embeddings = LocalEmbeddings(model_name='all-MiniLM-L6-v2')  # Use a local model


        self._setup_db()

    def _setup_db(self):
        """Ensure proper constraints and indexes exist."""
        try:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.index IS NOT NULL"
            ]
            for constraint in constraints:
                self.neo4j.run_query(constraint)
            
            # Create vector index

            #updated code here to match local embedding dimensions from 1536 to 384
            vector_index = """
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk)
            ON (c.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
            self.neo4j.run_query(vector_index)
            
        except Exception as e:
            logging.error(f"Error setting up database: {e}")
            raise

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


    ##### semantic / linking
    def create_semantic_relationships(self, doc_id: str):
        # Get all chunks for a document
        chunks = self.neo4j.run_query("""
            MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
            RETURN c.id as chunk_id, c.text as text, c.embedding as embedding
        """, {"doc_id": doc_id})
        
        # Find similar chunks using embeddings
        for chunk1 in chunks:
            for chunk2 in chunks:
                if chunk1['chunk_id'] != chunk2['chunk_id']:
                    similarity = self.calculate_similarity(
                        chunk1['embedding'], 
                        chunk2['embedding']
                    )
                    # If chunks are semantically similar, create relationship
                    if similarity > 0.8:  # Threshold for similarity
                        self.neo4j.run_query("""
                            MATCH (c1:Chunk {id: $chunk1_id})
                            MATCH (c2:Chunk {id: $chunk2_id})
                            MERGE (c1)-[:SEMANTICALLY_RELATED]->(c2)
                        """, {
                            "chunk1_id": chunk1['chunk_id'],
                            "chunk2_id": chunk2['chunk_id']
                        })


    def extract_and_link_entities(self, chunk_text: str, chunk_id: str):
        # Use OpenAI to extract entities
        prompt = f"Extract key entities (people, organizations, concepts) from this text: {chunk_text}"
        response = self.chat_model.predict(prompt)
        entities = parse_entities(response)  # Parse the response
        
        # Create entity nodes and relationships
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

    async def process_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> Dict:
            """Process document with enhanced structure."""
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
    
                # Create document with metadata
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
    
                # Process chunks in sequence
                for idx, chunk_text in enumerate(chunks):
                    chunk_id = f"{doc_id}-chunk-{idx}"
                    embedding = self.embeddings.embed_query(chunk_text)
                    
                    # Create chunk with all necessary properties
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
    
                    # If not first chunk, create NEXT relationship
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
    
                # Verify the structure
                verification = self.neo4j.run_query(
                    """
                    MATCH (d:Document {id: $doc_id})
                    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                    WITH d, count(c) as chunk_count
                    RETURN d.chunk_count = chunk_count as is_valid
                    """,
                    {"doc_id": doc_id}
                )
    
                if not verification[0].get('is_valid', False):
                    logging.warning(f"Document {doc_id} structure verification failed")
    
                return {
                    "status": "success",
                    "file": document_name,
                    "chunks_stored": len(chunks),
                    "doc_id": doc_id
                }
    
            except Exception as e:
                logging.error(f"Error processing document: {e}")
                # Cleanup on failure
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

    
    def similarity_search(self, query: str, limit: int = 5, context_window: int = 1) -> List[Dict[str, Any]]:
        """
        Enhanced similarity search with fixed traversal patterns.
        
        Args:
            query: Search query text
            limit: Number of primary results to return
            context_window: Number of adjacent chunks to include
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
                
                # Sort context chunks by relative position
                context = sorted(
                    result['context_chunks'], 
                    key=lambda x: x['relative_position']
                )
                
                # Format the result with context
                formatted_result = {
                    "chunk_id": result['chunk_id'],
                    "chunk_text": result['chunk_text'],
                    "score": result['similarity_score'],
                    "source": result['source_document'],
                    "index": result['chunk_index'],
                    "context": {
                        "before": [c['text'] for c in context if c['relative_position'] < 0],
                        "after": [c['text'] for c in context if c['relative_position'] > 0]
                    }
                }
                
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error in traversal search: {e}")
            # Fallback to basic search
            return self._basic_similarity_search(query, limit)
    
    def _basic_similarity_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fallback basic similarity search."""
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
            
            return [{
                "chunk_id": item["chunk_id"],
                "chunk_text": item["chunk_text"],
                "score": item["score"],
                "source": item["source"]
            } for item in results]
        except Exception as e:
            logging.error(f"Error in basic search: {e}")
            return []


    def verify_graph_structure(self) -> Dict[str, Any]:
            """Verify the entire graph structure."""
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

    async def query_knowledge(self, user_query: str) -> Dict:
        """Perform a RAG query with enhanced context."""
        try:
            # Get results without min_score parameter
            search_results = self.similarity_search(
                query=user_query,
                limit=5,
                context_window=2  # Get 2 chunks before/after
            )
            
            if not search_results:
                return {
                    "answer": "No relevant information found in the database.",
                    "search_results": []
                }
            
            # Create prompt with context
            context_sections = []
            for result in search_results:
                # Add main chunk with source
                section = f"\nSource: {result['source']}\nRelevance Score: {round(result['score'], 2)}\n"
                section += f"Content: {result['chunk_text']}\n"
                
                # Add context before/after if available
                if result['context']['before']:
                    section += "\nPrevious Context:\n" + "\n".join(result['context']['before'])
                if result['context']['after']:
                    section += "\nFollowing Context:\n" + "\n".join(result['context']['after'])
                
                context_sections.append(section)
            
            context_text = "\n---\n".join(context_sections)
            
            system_prompt = """You are a helpful assistant analyzing document content.
    Please use the provided context to answer the question accurately.
    Each section contains:
    - The main relevant text
    - Context from before and after (if available)
    - The source document and relevance score
    
    Use this information to provide a comprehensive answer. If the context doesn't fully answer the question, acknowledge the limitations.
    
    Context:
    {context}""".format(context=context_text)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
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
                            c["chunk_text"][:70] + "..." if len(c["chunk_text"]) > 70 else c["chunk_text"]
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
        # Set share=True to create a public link
        self.interface.launch(server_port=7860, share=True, inbrowser=True)

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    processor = GraphRAGProcessor()
    chat_app = ChatInterface(processor)
    chat_app.launch()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
