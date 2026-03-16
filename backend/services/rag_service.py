import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import logging

logger = logging.getLogger("rag_service")


class RAGService:

    def __init__(self):

        self.kb_path = Path(__file__).parent.parent / "data" / "knowledge_base.json"

        # Persistent ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Local embedding model
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="battery_docs",
            embedding_function=self.ef
        )

        if self.collection.count() == 0:
            self._load_and_persist_knowledge_base()
        else:
            logger.info(
                "RAG Service Ready: %d docs loaded in ChromaDB.",
                self.collection.count()
            )

    def _load_and_persist_knowledge_base(self):

        if not self.kb_path.exists():
            logger.warning("Knowledge base not found.")
            return

        with open(self.kb_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        logger.info("Loading %d docs into ChromaDB...", len(documents))

        ids = [doc.get("id") for doc in documents]
        texts = [doc.get("content", "") for doc in documents]
        metadatas = [{"title": doc.get("title", "Unknown")} for doc in documents]

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        logger.info("Knowledge Base persisted to ChromaDB.")

    def add_documents(self, documents: list, metadatas: list, ids: list):

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info("Added %d new documents to ChromaDB.", len(documents))

        except Exception as e:
            logger.error("Error adding documents: %s", e)

    async def rerank_results(self, query: str, results: list):

        """
        Use the LLM to re-rank retrieved documents for better relevance.
        """

        try:
            from services.model_client import model_client

            prompt = f"""
You are a document relevance evaluator.

User Question:
{query}

Documents:
{json.dumps(results, indent=2)}

Rank the documents by relevance to the user question.

Return JSON only:

{{
"ranked_indices": [0,1,2]
}}
"""

            response = await model_client.generate_async(prompt, task="text")

            start = response.find("{")
            end = response.rfind("}")

            if start != -1 and end != -1:
                data = json.loads(response[start:end+1])
                order = data.get("ranked_indices", [])

                ranked = []
                for i in order:
                    if i < len(results):
                        ranked.append(results[i])

                if ranked:
                    return ranked

            return results

        except Exception as e:
            logger.error("Rerank error: %s", e)
            return results

    async def search(self, query: str, top_k: int = 3):

        try:

            # Retrieve more candidates initially
            results = self.collection.query(
                query_texts=[query],
                n_results=8
            )

            formatted_results = []

            if results.get("ids"):

                for i in range(len(results["ids"][0])):

                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                    formatted_results.append(
                        {
                            "title": metadata.get("title", "Unknown"),
                            "content": results["documents"][0][i],
                        }
                    )

            if not formatted_results:
                return []

            # Re-rank with LLM
            ranked = await self.rerank_results(query, formatted_results)

            return ranked[:top_k]

        except Exception as e:
            logger.error("Search error: %s", e)
            return []


rag_service = RAGService()