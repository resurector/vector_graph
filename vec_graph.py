import os
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

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
        self.embeddings = OpenAIEmbeddings(
        model=self.config.openai_embedding_model,
        api_key=self.config.openai_api_key  # Explicitly pass API key
        )
        
        self._setup_db()

    def _setup_db(self):
        """Ensure constraints and vector index exist in Neo4j."""
        try:
            # Create constraints for Document and Chunk nodes
            constraint_queries = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            ]
            for cq in constraint_queries:
                self.neo4j.run_query(cq)
            
            # Updated vector index creation syntax
            vector_index_query = """
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk)
            ON (c.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
            try:
                self.neo4j.run_query(vector_index_query)
            except Exception as e:
                # Index might already exist, which is fine
                logging.info(f"Vector index creation note: {e}")
                
        except Exception as e:
            logging.error(f"Error setting up Neo4j constraints / indexes: {e}")

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

    async def process_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 100, batch_size: int = 10) -> Dict:
        """
        Process a document with batched embedding generation.
        
        Args:
            file_path: Path to the document
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Number of chunks to process in each embedding batch
        """
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
            
            # Generate document ID
            document_name = os.path.basename(file_path)
            unique_id = uuid.uuid4().hex
            doc_id = f"doc-{document_name}-{unique_id}"
            await self.create_semantic_relationships(doc_id)

            for chunk in chunks:
                await self.extract_and_link_entities(chunk.text, chunk.id)
    
            # Create Document node
            self.neo4j.run_query(
                """
                CREATE (d:Document {id: $doc_id})
                SET d.file_name = $file_name, 
                    d.upload_date = timestamp(),
                    d.chunk_count = $chunk_count
                """,
                {
                    "doc_id": doc_id, 
                    "file_name": document_name,
                    "chunk_count": len(chunks)
                }
            )
    
            # Process chunks in batches
            total_chunks = 0
            for batch_start in range(0, len(chunks), batch_size):
                # Get the current batch
                batch_end = min(batch_start + batch_size, len(chunks))
                current_batch = chunks[batch_start:batch_end]
                
                try:
                    # Generate embeddings for the entire batch
                    batch_embeddings = self.embeddings.embed_documents(current_batch)
                    
                    # Prepare batch data
                    batch_data = []
                    for idx, (chunk_text, embedding) in enumerate(zip(current_batch, batch_embeddings)):
                        global_idx = batch_start + idx
                        chunk_id = f"{doc_id}-chunk-{global_idx}"
                        
                        batch_data.append({
                            "chunk_id": chunk_id,
                            "chunk_text": chunk_text,
                            "embedding": embedding,
                            "idx": global_idx,
                            "doc_id": doc_id
                        })
                    
                    # Store all chunks in the batch
                    self.neo4j.run_query(
                        """
                        UNWIND $batch_data AS data
                        CREATE (c:Chunk {
                            id: data.chunk_id,
                            text: data.chunk_text,
                            embedding: data.embedding,
                            index: data.idx
                        })
                        WITH c, data
                        MATCH (d:Document {id: data.doc_id})
                        MERGE (d)-[:HAS_CHUNK]->(c)
                        """,
                        {"batch_data": batch_data}
                    )
                    
                    total_chunks += len(current_batch)
                    
                    # Log progress
                    progress = (batch_end / len(chunks)) * 100
                    logging.info(f"Processing document: {progress:.1f}% complete")
                    
                except Exception as e:
                    logging.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
                    # Continue with next batch instead of failing entirely
                    continue
            
            return {
                "status": "success",
                "file": document_name,
                "chunks_stored": total_chunks,
                "total_chunks": len(chunks)
            }
            
        except Exception as e:
            logging.error(f"Error processing document: {e}")
            # Attempt to clean up if document node was created
            try:
                self.neo4j.run_query(
                    """
                    MATCH (d:Document {id: $doc_id})
                    DETACH DELETE d
                    """,
                    {"doc_id": doc_id}
                )
            except:
                pass
            return {"status": "error", "message": str(e)}

    def similarity_search(self, query: str, limit: int = 5, context_window: int = 1) -> List[Dict[str, Any]]:
        """
        Enhanced similarity search with context awareness and relationship traversal.
        
        Args:
            query: The search query
            limit: Number of primary results to return
            context_window: Number of adjacent chunks to include before/after matches
        """
        query_embedding = self.embeddings.embed_query(query)
        
        # Enhanced vector search query with context and relationships
        vector_search_query = """
        // Initial vector similarity search
        CALL db.index.vector.queryNodes(
            'chunk_embeddings',
            $k,
            $query_embedding
        ) YIELD node, score
        WITH node, score
        
        // Get document information
        MATCH (doc:Document)-[:HAS_CHUNK]->(node)
        
        // Get surrounding context chunks
        MATCH (doc)-[:HAS_CHUNK]->(context:Chunk)
        WHERE context.index >= node.index - $window 
        AND context.index <= node.index + $window
        
        // Aggregate results
        WITH 
            node as main_chunk,
            score as similarity_score,
            doc.file_name as source_doc,
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
                vector_search_query,
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
                result = item['result']  # Neo4j wraps our result object
                
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
            logging.error(f"Error in similarity search: {e}")
            # Fallback to basic search if advanced query fails
            fallback_query = """
            CALL db.index.vector.queryNodes(
                'chunk_embeddings',
                $k,
                $query_embedding
            ) YIELD node, score
            RETURN 
                node.id AS chunk_id,
                node.text AS chunk_text,
                score
            """
            
            results = self.neo4j.run_query(
                fallback_query,
                {
                    "k": limit,
                    "query_embedding": query_embedding
                }
            )
            
            return [{
                "chunk_id": item["chunk_id"],
                "chunk_text": item["chunk_text"],
                "score": item["score"]
            } for item in results]



    

    async def query_knowledge(self, user_query: str) -> Dict:
        """
        Enhanced RAG query with better context gathering and structured output.
        """
        try:
            # Get relevant chunks with expanded context
            relevant_chunks = self.neo4j.run_query("""
                // First get similar chunks using vector search
                CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
                YIELD node, score
                WITH node, score
                
                // Get document info for context
                MATCH (doc:Document)-[:HAS_CHUNK]->(node)
                
                // Get adjacent chunks for additional context
                OPTIONAL MATCH (doc)-[:HAS_CHUNK]->(adjacent:Chunk)
                WHERE adjacent.index IN [node.index - 1, node.index + 1]
                
                RETURN 
                    node.id AS chunk_id,
                    node.text AS chunk_text,
                    node.index AS chunk_index,
                    score,
                    doc.file_name AS source_doc,
                    collect(adjacent.text) AS adjacent_chunks
            """, {
                "k": 5,
                "query_embedding": self.embeddings.embed_query(user_query)
            })
    
            if not relevant_chunks:
                return {
                    "answer": "No relevant information found in the database.",
                    "search_results": []
                }
    
            # Structure context with document source and adjacent chunks
            context_sections = []
            for chunk in relevant_chunks:
                # Format main chunk with source and score
                main_chunk = (
                    f"Source: {chunk['source_doc']}, Chunk {chunk['chunk_index']}\n"
                    f"Relevance: {round(chunk['score'], 2)}\n"
                    f"Content: {chunk['chunk_text']}"
                )
                context_sections.append(main_chunk)
                
                # Add adjacent chunks if available
                if chunk['adjacent_chunks']:
                    context_sections.append(
                        "Related Context:\n" + 
                        "\n".join(chunk['adjacent_chunks'])
                    )
    
            context_text = "\n\n---\n\n".join(context_sections)
    
            # Enhanced system prompt with specific instructions
            system_prompt = """You are a knowledgeable assistant analyzing provided document chunks.
            
    Guidelines:
    - Base your answer only on the provided context
    - If the context doesn't fully answer the question, acknowledge the limitations
    - Quote relevant parts of the context to support your answer
    - If you need to connect information from multiple chunks, explain how they relate
    - If there are contradictions in the context, point them out
    
    Context:
    {context}
    
    Remember to maintain accuracy and cite specific chunks when possible.""".format(context=context_text)
    
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
    
            # Generate response with slightly higher temperature for more natural answers
            response = await self.chat_model.agenerate(
                [[m for m in messages]], 
                temperature=0.3
            )
            answer_text = response.generations[0][0].text.strip()
    
            # Format search results for UI
            search_results = [{
                "chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["chunk_text"],
                "score": chunk["score"],
                "source": chunk["source_doc"],
                "index": chunk["chunk_index"]
            } for chunk in relevant_chunks]
    
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
