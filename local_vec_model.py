import os
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
import hashlib

import fitz  # PyMuPDF for PDF
import gradio as gr
from docx import Document as DocxDocument
from dotenv import load_dotenv

# If you still want to use OpenAI for the chat (LLM):
from langchain_openai import ChatOpenAI

# Other LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage

from neo4j import GraphDatabase

# ---- Local Embeddings (Sentence Transformers) ----
from sentence_transformers import SentenceTransformer

class LocalEmbeddings:
    """
    A simple class to generate embeddings locally using a SentenceTransformer model.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

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
        
        # Provider configuration (still used if you want OpenAI for chat)
        self.provider_type = os.getenv('PROVIDER_TYPE', 'openai')  # 'openai' or 'azure'
        
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

        self._validate()

    def _validate(self):
        # Basic checks
        if not self.neo4j_uri or not self.neo4j_username:
            raise ValueError("Missing Neo4j connection details")
        # Only require OpenAI key if you're actually using ChatOpenAI
        if self.provider_type == 'openai' and not self.openai_api_key:
           raise ValueError("OpenAI API key is required if using ChatOpenAI.")

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
        
        # Set environment variables for OpenAI (only needed if using ChatOpenAI)
        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
        
        # Create or open Neo4j connection
        self.neo4j = Neo4jHelper(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_username,
            password=self.config.neo4j_password
        )

         # Initialize OpenAI with caching
        self.chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            api_key=self.config.openai_api_key
        )
        
        # --- Use local embeddings instead of OpenAI embeddings ---
        self.embeddings = LocalEmbeddings(model_name='all-MiniLM-L6-v2')
        
        # Set up Neo4j constraints & indexes (including the 384-dim vector index)
        self._setup_db()

    def _setup_db(self):
        """Ensure proper constraints and indexes exist in Neo4j."""
        try:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.index IS NOT NULL"
            ]
            for constraint in constraints:
                self.neo4j.run_query(constraint)
            
            # Updated vector index for 384 dimensions (all-MiniLM-L6-v2)
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
    
    def _get_cached_response(self, query: str, max_age_hours: int = 24) -> Dict[str, Any]:
        """Check for semantically similar cached queries"""
        query_embedding = self.embeddings.embed_query(query)
        
        cache_query = """
        WITH datetime() as now
        MATCH (q:Query)
        WHERE duration.between(q.timestamp, now).hours < $max_age
        WITH q, gds.similarity.cosine(q.embedding, $embedding) as similarity
        WHERE similarity >= 0.95
        MATCH (q)-[:HAS_ANSWER]->(a:Answer)
        RETURN q.query as cached_query, a.text as answer, similarity
        ORDER BY similarity DESC
        LIMIT 1
        """
        
        result = self.neo4j.run_query(
            cache_query,
            {
                "embedding": query_embedding,
                "max_age": max_age_hours
            }
        )
        
        return result[0] if result else None
    
    def _store_in_cache(self, query: str, answer: str):
        """Store query and answer in semantic cache"""
        query_embedding = self.embeddings.embed_query(query)
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        
        cache_store = """
        CREATE (q:Query {
            hash: $hash,
            query: $query,
            embedding: $embedding,
            timestamp: datetime()
        })
        CREATE (a:Answer {
            id: $answer_id,
            text: $answer
        })
        CREATE (q)-[:HAS_ANSWER]->(a)
        """
        
        self.neo4j.run_query(
            cache_store,
            {
                "hash": query_hash,
                "query": query,
                "embedding": query_embedding,
                "answer_id": f"ans-{query_hash}",
                "answer": answer
            }
        )

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

    def calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Simple cosine similarity calculation.
        (Only needed if you do chunk-to-chunk comparisons in Python.)
        """
        import numpy as np
        v1 = np.array(emb1)
        v2 = np.array(emb2)
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return float(dot / (norm1 * norm2))

    async def process_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> Dict:
        """Process document, chunk it, store in Neo4j with local embeddings."""
        try:
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

            # Create Document node
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

            # Store chunks
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}-chunk-{idx}"
                embedding = self.embeddings.embed_query(chunk_text)

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

                # Link sequential chunks
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

            if not verification or not verification[0].get('is_valid', False):
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
        Search for chunks via vector similarity in Neo4j, then gather adjacent context chunks.
        """
        query_embedding = self.embeddings.embed_query(query)
        
        traversal_query = """
        // Vector similarity search using the 'chunk_embeddings' index
        CALL db.index.vector.queryNodes(
            'chunk_embeddings',
            $k,
            $query_embedding
        ) YIELD node, score

        MATCH (doc:Document)-[:HAS_CHUNK]->(node)

        // Get surrounding context
        MATCH (doc)-[:HAS_CHUNK]->(context:Chunk)
        WHERE context.index >= node.index - $window 
          AND context.index <= node.index + $window

        WITH 
            node as main_chunk,
            doc.file_name as source_doc,
            score as similarity_score,
            collect(DISTINCT {
                text: context.text,
                index: context.index,
                relative_position: context.index - node.index
            }) as context_chunks

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
                    "k": limit * 2,  # slightly overfetch
                    "limit": limit,
                    "window": context_window,
                    "query_embedding": query_embedding
                }
            )
            
            formatted_results = []
            for item in results:
                result = item['result']
                context = sorted(
                    result['context_chunks'],
                    key=lambda x: x['relative_position']
                )
                formatted_results.append({
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
                })
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error in traversal search: {e}")
            # fallback
            return self._basic_similarity_search(query, limit)

    

    def _basic_similarity_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fallback basic similarity search if the advanced query fails."""
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

    async def query_knowledge(self, user_query: str) -> Dict[str, Any]:
        """Enhanced RAG query with semantic caching and advanced context retrieval"""
        try:
            # Check cache first
            cached = self._get_cached_response(user_query)
            if cached:
                return {
                    "answer": cached["answer"],
                    "source": "cache",
                    "cached_query": cached["cached_query"],
                    "similarity": cached["similarity"]
                }

            # Enhanced similarity search with hybrid approach
            search_results = self._enhanced_similarity_search(
                query=user_query,
                limit=5,
                context_window=2
            )

            if not search_results:
                return {
                    "answer": "No relevant information found in the database.",
                    "search_results": []
                }

            # Build enhanced context with metadata
            context = self._build_enhanced_context(search_results, user_query)
            
            # Get answer from OpenAI
            response = await self._get_llm_response(context, user_query)
            
            # Store in cache
            self._store_in_cache(user_query, response)

            return {
                "answer": response,
                "search_results": search_results,
                "source": "fresh"
            }

        except Exception as e:
            logging.error(f"Query error: {e}")
            return {"answer": f"Error: {str(e)}", "search_results": []}

    def _format_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format raw Neo4j search results into a structured format.
        
        Args:
            results: List of raw search results from Neo4j query
            
        Returns:
            List of formatted results with chunk info and context
        """
        formatted_results = []
        
        for item in results:
            result = item['result']
            # Sort context chunks by their position relative to the main chunk
            context = sorted(
                result['context_chunks'],
                key=lambda x: x['relative_position']
            )
            
            formatted_results.append({
                "chunk_id": result['chunk_id'],
                "chunk_text": result['chunk_text'],
                "score": result['similarity_score'],
                "source": result['source_document'],
                "context": {
                    "before": [
                        c['text'] for c in context 
                        if c['relative_position'] < 0
                    ],
                    "after": [
                        c['text'] for c in context 
                        if c['relative_position'] > 0
                    ]
                }
            })
        
        return formatted_results

    def _enhanced_similarity_search(self, query: str, limit: int = 5, context_window: int = 2) -> List[Dict[str, Any]]:
        """Enhanced vector similarity search with hybrid retrieval"""
        query_embedding = self.embeddings.embed_query(query)
        
        hybrid_query = """
        // Vector similarity search
        CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
        YIELD node as chunk, score as vector_score
        
        // Get document info
        MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)
        
        // Get surrounding context
        MATCH (doc)-[:HAS_CHUNK]->(context:Chunk)
        WHERE context.index >= chunk.index - $window 
          AND context.index <= chunk.index + $window
        
        // Optional: Add text similarity boost
        WITH chunk, doc, vector_score, context,
             apoc.text.clean(chunk.text) as clean_text,
             apoc.text.clean($query) as clean_query
        WITH chunk, doc, vector_score, context,
             1.0 * vector_score + 
             CASE 
                WHEN clean_text CONTAINS clean_query THEN 0.2
                ELSE 0
             END as final_score
        
        // Collect and return results
        WITH chunk, doc, final_score,
             collect(DISTINCT {
                text: context.text,
                index: context.index,
                relative_position: context.index - chunk.index
             }) as context_chunks
        ORDER BY final_score DESC
        LIMIT $limit
        
        RETURN {
            chunk_id: chunk.id,
            chunk_text: chunk.text,
            similarity_score: final_score,
            source_document: doc.file_name,
            context_chunks: context_chunks
        } as result
        """
        
        try:
            results = self.neo4j.run_query(
                hybrid_query,
                {
                    "k": limit * 2,
                    "limit": limit,
                    "window": context_window,
                    "query_embedding": query_embedding,
                    "query": query
                }
            )
            
            return self._format_search_results(results)
            
        except Exception as e:
            logging.error(f"Enhanced search error: {e}")
            return self._basic_similarity_search(query, limit)

    def _build_enhanced_context(self, search_results: List[Dict[str, Any]], query: str) -> str:
        """Build rich context with metadata and relevance information"""
        context_sections = []
        
        for idx, result in enumerate(search_results, 1):
            section = (
                f"[Source {idx}: {result['source']} "
                f"(Relevance: {round(result['score'], 2)})]"
                f"\nContent: {result['chunk_text']}\n"
            )
            
            if result.get('context', {}).get('before'):
                section += "\nPrevious Context:\n" + "\n".join(
                    result['context']['before']
                )
            if result.get('context', {}).get('after'):
                section += "\nFollowing Context:\n" + "\n".join(
                    result['context']['after']
                )
                
            context_sections.append(section)

        separator = '-' * 40
        formatted_sections = '\n' + separator + '\n'
        formatted_sections += ('\n' + separator + '\n').join(context_sections)
        
        return '\n'.join([
            "Query Analysis:",
            f"User Question: {query}",
            "",
            "Relevant Document Sections:",
            separator,
            formatted_sections
        ])

    async def _get_llm_response(self, context: str, query: str) -> str:
        """Get response from OpenAI with optimized prompting"""
        system_prompt = (
            "You are a knowledgeable assistant analyzing document content.\n"
            "Provide accurate, concise answers based on the provided context.\n"
            "If the context doesn't fully answer the question, acknowledge the limitation.\n"
            "Focus on the most relevant information and maintain factual accuracy."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]

        response = await self.chat_model.agenerate([[m for m in messages]])
        return response.generations[0][0].text.strip()


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
                            (
                                c["chunk_text"][:70] + "..." 
                                if len(c["chunk_text"]) > 70 
                                else c["chunk_text"]
                            )
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
                chunk_size_slider = gr.Slider(
                    100, 2000, step=100, value=500, label="Chunk Size"
                )
                overlap_slider = gr.Slider(
                    0, 500, step=50, value=100, label="Chunk Overlap"
                )
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
                        return (
                            f"Processed '{result['file']}' "
                            f"with {result['chunks_stored']} chunks stored."
                        )
                    else:
                        return f"Error: {result['message']}"

                process_button.click(
                    fn=process_handler,
                    inputs=[file_input, chunk_size_slider, overlap_slider],
                    outputs=[status_box]
                )

        return demo

    def launch(self):
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
