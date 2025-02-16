import os
import logging
import asyncio
import uuid
import hashlib
from typing import Optional, List, Dict, Any

import fitz  # PyMuPDF for PDF
import gradio as gr
from docx import Document as DocxDocument
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer



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
# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
class Config:
    """
    Holds configuration for Neo4j connection and OpenAI usage (if needed).
    Loads values from .env or environment variables.
    """
    def __init__(self):
        load_dotenv()

        # Neo4j connection details
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', '')

        # OpenAI-specific
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

        self._validate()

    def _validate(self):
        """
        Basic validation of required variables.
        """
        if not self.neo4j_uri or not self.neo4j_username:
            raise ValueError("Missing Neo4j connection details")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for ChatOpenAI usage.")


# ------------------------------------------------------------------------------
# Neo4j Helper
# ------------------------------------------------------------------------------
class Neo4jHelper:
    """
    Small helper class to handle Neo4j connections & queries.
    """
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: Optional[Dict] = None):
        """
        Run a Cypher query with optional parameters.
        Returns the result in .data() form.
        """
        with self.driver.session() as session:
            return session.run(query, parameters or {}).data()


# ------------------------------------------------------------------------------
# Local Embeddings (Sentence Transformers)
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# Main RAG Processor
# ------------------------------------------------------------------------------
class GraphRAGProcessor:
    """
    Main class that orchestrates:
      1. Document ingestion & chunking
      2. Vector storage in Neo4j
      3. Query caching and retrieval
      4. Retrieval-Augmented Generation with an LLM
    """
    def __init__(self):
        self.config = Config()
        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key

        self.neo4j = Neo4jHelper(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_username,
            password=self.config.neo4j_password
        )

        # Chat model (OpenAI)
        self.chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            api_key=self.config.openai_api_key
        )

        self.embeddings = LocalEmbeddings(model_name='all-MiniLM-L6-v2')

        # Ensure DB constraints and indexes
        self._setup_db()

    def _setup_db(self):
        """
        Sets up Neo4j constraints and a vector index for chunk embeddings.
        """
        try:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.index IS NOT NULL"
            ]
            for cql in constraints:
                self.neo4j.run_query(cql)

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
        """
        Extract text from PDF, DOCX, or TXT files.
        """
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

    async def process_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> Dict:
        """
        Process a document by splitting into chunks, embedding, and storing in Neo4j.
        """
        try:
            text = self.extract_text_from_file(file_path)
            if not text.strip():
                return {"status": "error", "message": "Empty file or text extraction failed."}

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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

            # Create Chunk nodes and link them
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

                # Link to previous chunk for adjacency
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

            # Optional verification
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

    def _get_cached_response(self, query: str, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Check for a semantically similar cached query (within the last `max_age_hours`).
        """
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
            cache_query, {"embedding": query_embedding, "max_age": max_age_hours}
        )
        return result[0] if result else None

    def _store_in_cache(self, query: str, answer: str):
        """
        Store query & answer in a semantic cache for future lookups.
        """
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

    async def query_knowledge(self, user_query: str) -> Dict[str, Any]:
        """
        Query the knowledge base with optional caching.
        If no valid cache found, perform an enhanced similarity search, then call LLM.
        """
        try:
            # Check cache
            cached = self._get_cached_response(user_query)
            if cached:
                return {
                    "answer": cached["answer"],
                    "source": "cache",
                    "cached_query": cached["cached_query"],
                    "similarity": cached["similarity"]
                }

            # Not in cache → do retrieval + LLM
            search_results = self._enhanced_similarity_search(user_query, limit=5, context_window=2)
            if not search_results:
                return {"answer": "No relevant information found.", "search_results": []}

            context = self._build_enhanced_context(search_results, user_query)
            response = await self._get_llm_response(context, user_query)

            # Cache new result
            self._store_in_cache(user_query, response)

            return {
                "answer": response,
                "search_results": search_results,
                "source": "fresh"
            }

        except Exception as e:
            logging.error(f"Query error: {e}")
            return {"answer": f"Error: {str(e)}", "search_results": []}

    def _enhanced_similarity_search(self, query: str, limit: int = 5, context_window: int = 2) -> List[Dict[str, Any]]:
        """
        Enhanced vector similarity search in Neo4j, with a small text-based similarity boost.
        """
        query_embedding = self.embeddings.embed_query(query)
        hybrid_query = """
        // Vector similarity search
        CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
        YIELD node as chunk, score as vector_score

        MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)

        // Gather surrounding context
        MATCH (doc)-[:HAS_CHUNK]->(context:Chunk)
        WHERE context.index >= chunk.index - $window 
          AND context.index <= chunk.index + $window

        WITH chunk, doc, vector_score, context,
             apoc.text.clean(chunk.text) as clean_text,
             apoc.text.clean($query) as clean_query

        WITH chunk, doc, vector_score, context,
             1.0 * vector_score +
             CASE
                WHEN clean_text CONTAINS clean_query THEN 0.2
                ELSE 0
             END as final_score

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
            # Fallback
            return self._basic_similarity_search(query, limit)

    def _basic_similarity_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback to a simpler vector similarity search if advanced search fails.
        """
        query_embedding = self.embeddings.embed_query(query)
        basic_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
        YIELD node, score
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
                {"k": limit, "limit": limit, "query_embedding": query_embedding}
            )
            return [
                {
                    "chunk_id": r["chunk_id"],
                    "chunk_text": r["chunk_text"],
                    "score": r["score"],
                    "source": r["source"],
                    "context": {"before": [], "after": []}  # Minimal
                }
                for r in results
            ]
        except Exception as e:
            logging.error(f"Error in basic search: {e}")
            return []

    def _format_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format raw Neo4j search results into a structured format with chunk context.
        """
        formatted = []
        for row in results:
            item = row['result']
            context_chunks = sorted(
                item['context_chunks'],
                key=lambda x: x['relative_position']
            )
            formatted.append({
                "chunk_id": item["chunk_id"],
                "chunk_text": item["chunk_text"],
                "score": item["similarity_score"],
                "source": item["source_document"],
                "context": {
                    "before": [c['text'] for c in context_chunks if c['relative_position'] < 0],
                    "after":  [c['text'] for c in context_chunks if c['relative_position'] > 0]
                }
            })
        return formatted

    def _build_enhanced_context(self, search_results: List[Dict[str, Any]], query: str) -> str:
        """
        Build a textual context string from the retrieved chunks plus metadata
        (relevance score, source, etc.).
        """
        sections = []
        separator = '-' * 40

        for i, res in enumerate(search_results, 1):
            section = (
                f"[Source {i}: {res['source']} (Relevance: {round(res['score'], 2)})]\n"
                f"Content: {res['chunk_text']}\n"
            )
            before = res['context'].get('before', [])
            after = res['context'].get('after', [])
            if before:
                section += "\nPrevious Context:\n" + "\n".join(before)
            if after:
                section += "\nFollowing Context:\n" + "\n".join(after)
            sections.append(section)

        # Combine everything
        context_str = (
            f"Query Analysis:\nUser Question: {query}\n\n"
            f"Relevant Document Sections:\n{separator}\n"
            + f"{separator}\n".join(sections)
        )
        return context_str


    async def _get_llm_response(self, context: str, query: str) -> str:
        """
        Send a prompt to OpenAI Chat with the provided context and user query.
        """
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


# ------------------------------------------------------------------------------
# Gradio Chat Interface
# ------------------------------------------------------------------------------
class ChatInterface:
    """
    A simple Gradio interface that exposes:
      1. A chat tab to query the knowledge base
      2. A document ingestion tab
    """
    def __init__(self, processor: GraphRAGProcessor):
        self.processor = processor
        self.interface = self._setup_gradio()

    def _setup_gradio(self) -> gr.Blocks:
        with gr.Blocks(title="GraphRAG", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# GraphRAG: Ask Your Documents in Neo4j")

            # ----------- Chat Tab -----------
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

            # ----------- Process Document Tab -----------
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
        """
        Launch the Gradio interface.
        """
        self.interface.launch(server_port=7860, share=True, inbrowser=True)


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def main():
    processor = GraphRAGProcessor()
    chat_app = ChatInterface(processor)
    chat_app.launch()
    logging.info("GraphRAG interface launched.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
