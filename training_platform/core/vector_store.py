"""
Milvus Vector Store Integration

Provides vector storage for:
- Training data embeddings (for similarity search)
- Model outputs for evaluation
- Document retrieval for RAG-based training
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Milvus imports (optional)
try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
        MilvusClient,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("pymilvus not installed. Vector store features will be disabled.")

# Embedding client (using Dashscope/Aliyun)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai package not installed. Embedding features will be disabled.")


@dataclass
class VectorStoreConfig:
    """Configuration for Milvus vector store"""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "training_data"
    embedding_dim: int = 1024
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 128


class EmbeddingClient:
    """
    Embedding client using Dashscope text-embedding-v4.

    Can be swapped with other embedding providers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
    ):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai package is required for embeddings")

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is required")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )
        self.model = model
        self.dimensions = dimensions

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed multiple texts in batches"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            embeddings.extend([d.embedding for d in response.data])

        return embeddings


class MilvusVectorStore:
    """
    Milvus vector store for training data management.

    Features:
    - Store and retrieve training examples
    - Similarity search for data curation
    - Hybrid search (vector + filters)
    """

    def __init__(self, config: VectorStoreConfig = None):
        if not MILVUS_AVAILABLE:
            raise RuntimeError("pymilvus is required for vector store")

        self.config = config or VectorStoreConfig()
        self._connected = False
        self._collection: Optional[Collection] = None

    def connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port,
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {self.config.host}:{self.config.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def disconnect(self):
        """Disconnect from Milvus"""
        if self._connected:
            connections.disconnect("default")
            self._connected = False

    def _ensure_collection(self):
        """Ensure collection exists"""
        if not self._connected:
            self.connect()

        if utility.has_collection(self.config.collection_name):
            self._collection = Collection(self.config.collection_name)
            self._collection.load()
        else:
            self._create_collection()

    def _create_collection(self):
        """Create collection with schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Training data embeddings",
        )

        self._collection = Collection(
            name=self.config.collection_name,
            schema=schema,
        )

        # Create index
        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": {"nlist": self.config.nlist},
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        self._collection.load()

        logger.info(f"Created collection: {self.config.collection_name}")

    def insert(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]] = None,
        source: str = "unknown",
    ):
        """
        Insert vectors into the collection.

        Args:
            ids: Unique identifiers
            texts: Original text content
            embeddings: Vector embeddings
            metadata: Additional metadata for each entry
            source: Data source identifier
        """
        self._ensure_collection()

        import time
        now = int(time.time())

        data = [
            ids,
            texts,
            embeddings,
            metadata or [{} for _ in ids],
            [source] * len(ids),
            [now] * len(ids),
        ]

        self._collection.insert(data)
        self._collection.flush()

        logger.info(f"Inserted {len(ids)} vectors into {self.config.collection_name}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[str] = None,
        output_fields: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional filter expression
            output_fields: Fields to include in results

        Returns:
            List of matching documents with scores
        """
        self._ensure_collection()

        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": 10},
        }

        output_fields = output_fields or ["id", "text", "metadata", "source"]

        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=output_fields,
        )

        documents = []
        for hits in results:
            for hit in hits:
                doc = {
                    "id": hit.id,
                    "score": hit.score,
                }
                for field in output_fields:
                    if hasattr(hit.entity, field):
                        doc[field] = getattr(hit.entity, field)
                documents.append(doc)

        return documents

    def hybrid_search(
        self,
        query_embedding: List[float],
        keyword_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and filters.

        Args:
            query_embedding: Query vector
            keyword_filter: Text filter expression
            source_filter: Filter by source
            top_k: Number of results

        Returns:
            Filtered and ranked results
        """
        filters = []
        if source_filter:
            filters.append(f'source == "{source_filter}"')
        if keyword_filter:
            filters.append(keyword_filter)

        filter_expr = " and ".join(filters) if filters else None

        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filter_expr,
        )

    def delete(self, ids: List[str]):
        """Delete vectors by ID"""
        self._ensure_collection()
        expr = f'id in {ids}'
        self._collection.delete(expr)
        logger.info(f"Deleted {len(ids)} vectors")

    def count(self) -> int:
        """Get total vector count"""
        self._ensure_collection()
        return self._collection.num_entities

    def drop_collection(self):
        """Drop the collection"""
        if utility.has_collection(self.config.collection_name):
            utility.drop_collection(self.config.collection_name)
            logger.info(f"Dropped collection: {self.config.collection_name}")


class DatasetManager:
    """
    High-level dataset management using vector store.

    Features:
    - Load and embed training data
    - Similarity-based data curation
    - Deduplication
    - Data quality analysis
    """

    def __init__(
        self,
        vector_store: MilvusVectorStore = None,
        embedding_client: EmbeddingClient = None,
    ):
        self.vector_store = vector_store or MilvusVectorStore()
        self.embedding_client = embedding_client

    def _get_embedding_client(self) -> EmbeddingClient:
        """Get or create embedding client"""
        if self.embedding_client is None:
            self.embedding_client = EmbeddingClient()
        return self.embedding_client

    def load_jsonl(
        self,
        file_path: str,
        text_field: str = "text",
        id_field: str = "id",
        batch_size: int = 100,
        source: str = None,
    ) -> int:
        """
        Load and embed data from JSONL file.

        Args:
            file_path: Path to JSONL file
            text_field: Field containing text to embed
            id_field: Field containing unique ID
            batch_size: Batch size for embedding
            source: Source identifier (defaults to filename)

        Returns:
            Number of records loaded
        """
        import uuid
        from pathlib import Path

        source = source or Path(file_path).stem
        client = self._get_embedding_client()

        records_loaded = 0
        batch_ids = []
        batch_texts = []
        batch_metadata = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                text = record.get(text_field, "")
                if not text:
                    continue

                record_id = record.get(id_field) or str(uuid.uuid4())[:8]
                batch_ids.append(record_id)
                batch_texts.append(text[:8000])  # Truncate long texts
                batch_metadata.append({k: v for k, v in record.items()
                                       if k not in [text_field, id_field]})

                if len(batch_ids) >= batch_size:
                    self._insert_batch(batch_ids, batch_texts, batch_metadata, source, client)
                    records_loaded += len(batch_ids)
                    batch_ids, batch_texts, batch_metadata = [], [], []
                    logger.info(f"Loaded {records_loaded} records...")

            # Insert remaining
            if batch_ids:
                self._insert_batch(batch_ids, batch_texts, batch_metadata, source, client)
                records_loaded += len(batch_ids)

        logger.info(f"Finished loading {records_loaded} records from {file_path}")
        return records_loaded

    def _insert_batch(
        self,
        ids: List[str],
        texts: List[str],
        metadata: List[Dict],
        source: str,
        client: EmbeddingClient,
    ):
        """Insert a batch of records"""
        embeddings = client.embed_batch(texts)
        self.vector_store.insert(
            ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadata=metadata,
            source=source,
        )

    def find_similar(
        self,
        query: str,
        top_k: int = 10,
        source: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Find similar training examples.

        Args:
            query: Query text
            top_k: Number of results
            source: Filter by source

        Returns:
            Similar documents with scores
        """
        client = self._get_embedding_client()
        query_embedding = client.embed_text(query)

        return self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            source_filter=source,
            top_k=top_k,
        )

    def find_duplicates(
        self,
        threshold: float = 0.95,
        sample_size: int = 1000,
        batch_size: int = 100,
        source: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Find potential duplicate entries using vector similarity.

        This method samples records and finds pairs with similarity > threshold.

        Args:
            threshold: Similarity threshold for duplicates (0-1, cosine similarity)
            sample_size: Number of samples to check for duplicates
            batch_size: Batch size for processing
            source: Optional filter by data source

        Returns:
            List of duplicate pairs: [{"id1": ..., "id2": ..., "similarity": ..., "text1": ..., "text2": ...}]
        """
        duplicates = []
        seen_pairs = set()

        # Get samples to check
        samples = self._get_samples(sample_size, source)
        if not samples:
            logger.info("No samples found for duplicate detection")
            return []

        logger.info(f"Checking {len(samples)} samples for duplicates (threshold={threshold})")

        # For each sample, find similar items
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{len(samples)}")

            sample_id = sample.get('id')
            embedding = sample.get('embedding')

            if not embedding:
                continue

            # Search for similar items
            similar_items = self.vector_store.search(
                query_embedding=embedding,
                top_k=10,  # Check top 10 similar items
                output_fields=["id", "text", "metadata", "source"],
            )

            for item in similar_items:
                item_id = item.get('id')
                similarity = item.get('score', 0)

                # Skip self-matches
                if item_id == sample_id:
                    continue

                # Check threshold
                if similarity < threshold:
                    continue

                # Create ordered pair to avoid duplicates
                pair = tuple(sorted([sample_id, item_id]))
                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)
                duplicates.append({
                    "id1": sample_id,
                    "id2": item_id,
                    "similarity": round(similarity, 4),
                    "text1": sample.get('text', '')[:500],  # Truncate for display
                    "text2": item.get('text', '')[:500],
                    "source1": sample.get('source', ''),
                    "source2": item.get('source', ''),
                })

        # Sort by similarity descending
        duplicates.sort(key=lambda x: x['similarity'], reverse=True)

        logger.info(f"Found {len(duplicates)} duplicate pairs")
        return duplicates

    def _get_samples(
        self,
        sample_size: int,
        source: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Get random samples from the collection for duplicate detection.

        Args:
            sample_size: Number of samples to retrieve
            source: Optional filter by source

        Returns:
            List of sample records with embeddings
        """
        self.vector_store._ensure_collection()

        # Build filter expression
        expr = None
        if source:
            expr = f'source == "{source}"'

        try:
            # Query samples with embeddings
            # Note: Milvus doesn't have random sampling, so we query all and sample
            collection = self.vector_store._collection

            # Get total count
            total = collection.num_entities
            if total == 0:
                return []

            # Limit sample size to available records
            actual_sample_size = min(sample_size, total)

            # Query records - for large collections, we'd use iterator/pagination
            # For now, query with limit
            results = collection.query(
                expr=expr or "id != ''",  # Match all
                output_fields=["id", "text", "embedding", "metadata", "source"],
                limit=actual_sample_size,
            )

            return results
        except Exception as e:
            logger.error(f"Error getting samples: {e}")
            return []

    def delete_duplicates(
        self,
        duplicate_ids: List[str],
    ) -> int:
        """
        Delete duplicate records by ID.

        Args:
            duplicate_ids: List of IDs to delete

        Returns:
            Number of records deleted
        """
        if not duplicate_ids:
            return 0

        self.vector_store.delete(duplicate_ids)
        logger.info(f"Deleted {len(duplicate_ids)} duplicate records")
        return len(duplicate_ids)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        total_count = self.vector_store.count()
        return {
            "total_records": total_count,
            "collection_name": self.vector_store.config.collection_name,
        }


# Convenience functions

def get_vector_store(config: VectorStoreConfig = None) -> MilvusVectorStore:
    """Get vector store instance"""
    return MilvusVectorStore(config)


def get_dataset_manager(
    milvus_host: str = "localhost",
    milvus_port: int = 19530,
) -> DatasetManager:
    """Get dataset manager instance"""
    config = VectorStoreConfig(host=milvus_host, port=milvus_port)
    vector_store = MilvusVectorStore(config)
    return DatasetManager(vector_store=vector_store)
