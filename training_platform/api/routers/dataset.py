"""
Dataset API Router

Provides endpoints for dataset management using Milvus vector store:
- Upload and embed training data
- Similarity search for data curation
- Semantic deduplication
- Dataset statistics
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os
import uuid
import json
import tempfile
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Datasets"])

# Store for background deduplication tasks
_dedup_tasks: Dict[str, Dict[str, Any]] = {}


class LoadDatasetRequest(BaseModel):
    """Request to load a dataset"""
    file_path: str
    text_field: str = "text"
    id_field: str = "id"
    source_name: Optional[str] = None
    batch_size: int = 100


class SearchRequest(BaseModel):
    """Request for similarity search"""
    query: str
    top_k: int = 10
    source_filter: Optional[str] = None


class SearchResult(BaseModel):
    """Search result"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = {}


class DatasetStats(BaseModel):
    """Dataset statistics"""
    total_records: int
    collection_name: str
    sources: List[str] = []


def _get_dataset_manager():
    """Get or create dataset manager"""
    try:
        from ...core.vector_store import get_dataset_manager, MILVUS_AVAILABLE

        if not MILVUS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Milvus is not installed. Install pymilvus to use dataset features."
            )

        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

        return get_dataset_manager(milvus_host=milvus_host, milvus_port=milvus_port)
    except Exception as e:
        logger.error(f"Failed to get dataset manager: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/load")
async def load_dataset(request: LoadDatasetRequest) -> Dict[str, Any]:
    """
    Load and embed a JSONL dataset into Milvus.

    The file should be a JSONL file where each line is a JSON object
    with at least a text field.
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    try:
        manager = _get_dataset_manager()

        count = manager.load_jsonl(
            file_path=request.file_path,
            text_field=request.text_field,
            id_field=request.id_field,
            batch_size=request.batch_size,
            source=request.source_name,
        )

        return {
            "success": True,
            "records_loaded": count,
            "source": request.source_name or os.path.basename(request.file_path),
        }
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=List[SearchResult])
async def search_similar(request: SearchRequest) -> List[SearchResult]:
    """
    Search for similar training examples.

    Returns the most similar documents based on embedding similarity.
    """
    try:
        manager = _get_dataset_manager()

        results = manager.find_similar(
            query=request.query,
            top_k=request.top_k,
            source=request.source_filter,
        )

        return [
            SearchResult(
                id=r["id"],
                text=r.get("text", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=DatasetStats)
async def get_stats() -> DatasetStats:
    """
    Get dataset statistics.
    """
    try:
        manager = _get_dataset_manager()
        stats = manager.get_statistics()

        return DatasetStats(
            total_records=stats.get("total_records", 0),
            collection_name=stats.get("collection_name", "unknown"),
            sources=[],
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collection")
async def drop_collection() -> Dict[str, Any]:
    """
    Drop the dataset collection.

    WARNING: This will delete all data in the collection.
    """
    try:
        manager = _get_dataset_manager()
        manager.vector_store.drop_collection()

        return {"success": True, "message": "Collection dropped"}
    except Exception as e:
        logger.error(f"Failed to drop collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def check_health() -> Dict[str, Any]:
    """
    Check if Milvus connection is healthy.
    """
    try:
        from ...core.vector_store import MILVUS_AVAILABLE, OPENAI_AVAILABLE

        status = {
            "milvus_available": MILVUS_AVAILABLE,
            "embedding_available": OPENAI_AVAILABLE,
            "milvus_host": os.getenv("MILVUS_HOST", "localhost"),
            "milvus_port": os.getenv("MILVUS_PORT", "19530"),
        }

        if MILVUS_AVAILABLE:
            try:
                manager = _get_dataset_manager()
                manager.vector_store.connect()
                status["milvus_connected"] = True
            except Exception as e:
                status["milvus_connected"] = False
                status["milvus_error"] = str(e)

        return status
    except Exception as e:
        return {"error": str(e)}


# ============== File Upload Endpoints ==============

class UploadResponse(BaseModel):
    """Response for file upload"""
    success: bool
    task_id: str
    filename: str
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    text_field: str = Query("text", description="Field containing text to embed"),
    id_field: str = Query("id", description="Field containing unique ID"),
    source_name: Optional[str] = Query(None, description="Source name for the dataset"),
) -> UploadResponse:
    """
    Upload and process a JSONL file.

    The file is uploaded, saved temporarily, and processed in the background.
    Use the task_id to check processing status.
    """
    # Validate file type
    if not file.filename.endswith(('.jsonl', '.json', '.ndjson')):
        raise HTTPException(
            status_code=400,
            detail="Only .jsonl, .json, or .ndjson files are supported"
        )

    # Generate task ID
    task_id = str(uuid.uuid4())[:8]
    source = source_name or file.filename.rsplit('.', 1)[0]

    # Save file temporarily
    try:
        content = await file.read()
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"dataset_{task_id}.jsonl")

        with open(temp_path, 'wb') as f:
            f.write(content)

        # Initialize task status
        _dedup_tasks[task_id] = {
            "type": "upload",
            "status": "processing",
            "filename": file.filename,
            "source": source,
            "started_at": datetime.utcnow().isoformat(),
            "records_processed": 0,
            "error": None,
        }

        # Process in background
        background_tasks.add_task(
            _process_upload,
            task_id=task_id,
            file_path=temp_path,
            text_field=text_field,
            id_field=id_field,
            source=source,
        )

        return UploadResponse(
            success=True,
            task_id=task_id,
            filename=file.filename,
            message="File uploaded, processing in background",
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_upload(
    task_id: str,
    file_path: str,
    text_field: str,
    id_field: str,
    source: str,
):
    """Background task to process uploaded file"""
    try:
        manager = _get_dataset_manager()

        count = manager.load_jsonl(
            file_path=file_path,
            text_field=text_field,
            id_field=id_field,
            source=source,
        )

        _dedup_tasks[task_id].update({
            "status": "completed",
            "records_processed": count,
            "completed_at": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        _dedup_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat(),
        })

    finally:
        # Clean up temp file
        try:
            os.remove(file_path)
        except:
            pass


# ============== Deduplication Endpoints ==============

class DeduplicateRequest(BaseModel):
    """Request for deduplication"""
    threshold: float = Field(0.95, ge=0.5, le=1.0, description="Similarity threshold")
    sample_size: int = Field(1000, ge=10, le=10000, description="Number of samples to check")
    source_filter: Optional[str] = Field(None, description="Filter by source")


class DuplicatePair(BaseModel):
    """A pair of duplicate records"""
    id1: str
    id2: str
    similarity: float
    text1: str
    text2: str
    source1: str = ""
    source2: str = ""


class DeduplicateResponse(BaseModel):
    """Response for deduplication"""
    task_id: str
    status: str
    message: str


class DeduplicateStatusResponse(BaseModel):
    """Status of deduplication task"""
    task_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    duplicates_found: int = 0
    duplicates: List[DuplicatePair] = []
    error: Optional[str] = None


@router.post("/deduplicate", response_model=DeduplicateResponse)
async def start_deduplication(
    request: DeduplicateRequest,
    background_tasks: BackgroundTasks,
) -> DeduplicateResponse:
    """
    Start a deduplication task.

    This runs in the background. Use the task_id to check status
    and retrieve duplicate pairs for review.
    """
    task_id = str(uuid.uuid4())[:8]

    # Initialize task
    _dedup_tasks[task_id] = {
        "type": "deduplicate",
        "status": "processing",
        "started_at": datetime.utcnow().isoformat(),
        "threshold": request.threshold,
        "sample_size": request.sample_size,
        "source_filter": request.source_filter,
        "duplicates_found": 0,
        "duplicates": [],
        "error": None,
    }

    # Run in background
    background_tasks.add_task(
        _run_deduplication,
        task_id=task_id,
        threshold=request.threshold,
        sample_size=request.sample_size,
        source=request.source_filter,
    )

    return DeduplicateResponse(
        task_id=task_id,
        status="processing",
        message="Deduplication started in background",
    )


async def _run_deduplication(
    task_id: str,
    threshold: float,
    sample_size: int,
    source: Optional[str],
):
    """Background task for deduplication"""
    try:
        manager = _get_dataset_manager()

        duplicates = manager.find_duplicates(
            threshold=threshold,
            sample_size=sample_size,
            source=source,
        )

        _dedup_tasks[task_id].update({
            "status": "completed",
            "duplicates_found": len(duplicates),
            "duplicates": duplicates,
            "completed_at": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        _dedup_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat(),
        })


@router.get("/deduplicate/{task_id}", response_model=DeduplicateStatusResponse)
async def get_deduplication_status(task_id: str) -> DeduplicateStatusResponse:
    """
    Get the status of a deduplication task.

    Returns the list of duplicate pairs when completed.
    """
    if task_id not in _dedup_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _dedup_tasks[task_id]

    return DeduplicateStatusResponse(
        task_id=task_id,
        status=task.get("status", "unknown"),
        started_at=task.get("started_at", ""),
        completed_at=task.get("completed_at"),
        duplicates_found=task.get("duplicates_found", 0),
        duplicates=[DuplicatePair(**d) for d in task.get("duplicates", [])],
        error=task.get("error"),
    )


class DeleteDuplicatesRequest(BaseModel):
    """Request to delete duplicates"""
    ids: List[str] = Field(..., description="IDs of records to delete")


@router.post("/deduplicate/delete")
async def delete_duplicates(request: DeleteDuplicatesRequest) -> Dict[str, Any]:
    """
    Delete specified duplicate records.

    After reviewing duplicates, use this endpoint to delete the ones
    you want to remove.
    """
    if not request.ids:
        raise HTTPException(status_code=400, detail="No IDs provided")

    try:
        manager = _get_dataset_manager()
        deleted = manager.delete_duplicates(request.ids)

        return {
            "success": True,
            "deleted": deleted,
            "message": f"Deleted {deleted} records",
        }

    except Exception as e:
        logger.error(f"Delete duplicates failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Task Management Endpoints ==============

@router.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """
    List all background tasks (uploads and deduplication).
    """
    return {
        "tasks": [
            {
                "task_id": task_id,
                "type": task.get("type"),
                "status": task.get("status"),
                "started_at": task.get("started_at"),
                "completed_at": task.get("completed_at"),
            }
            for task_id, task in _dedup_tasks.items()
        ],
        "total": len(_dedup_tasks),
    }


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get status of a specific task.
    """
    if task_id not in _dedup_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _dedup_tasks[task_id]

    # Return different info based on task type
    result = {
        "task_id": task_id,
        "type": task.get("type"),
        "status": task.get("status"),
        "started_at": task.get("started_at"),
        "completed_at": task.get("completed_at"),
        "error": task.get("error"),
    }

    if task.get("type") == "upload":
        result["records_processed"] = task.get("records_processed", 0)
        result["filename"] = task.get("filename")
        result["source"] = task.get("source")
    elif task.get("type") == "deduplicate":
        result["threshold"] = task.get("threshold")
        result["duplicates_found"] = task.get("duplicates_found", 0)

    return result


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> Dict[str, Any]:
    """
    Delete a completed task from the task list.
    """
    if task_id not in _dedup_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _dedup_tasks[task_id]
    if task.get("status") == "processing":
        raise HTTPException(status_code=400, detail="Cannot delete a running task")

    del _dedup_tasks[task_id]
    return {"success": True, "message": "Task deleted"}


# ============== Batch Operations ==============

class BatchSearchRequest(BaseModel):
    """Request for batch similarity search"""
    queries: List[str] = Field(..., description="List of queries")
    top_k: int = Field(5, ge=1, le=100, description="Results per query")
    source_filter: Optional[str] = None


@router.post("/search/batch")
async def batch_search(request: BatchSearchRequest) -> Dict[str, Any]:
    """
    Perform similarity search for multiple queries.

    Useful for finding similar data across multiple prompts.
    """
    if len(request.queries) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 queries allowed")

    try:
        manager = _get_dataset_manager()
        results = {}

        for query in request.queries:
            similar = manager.find_similar(
                query=query,
                top_k=request.top_k,
                source=request.source_filter,
            )
            results[query] = [
                {
                    "id": r["id"],
                    "text": r.get("text", "")[:500],
                    "score": r.get("score", 0.0),
                }
                for r in similar
            ]

        return {
            "success": True,
            "results": results,
            "total_queries": len(request.queries),
        }

    except Exception as e:
        logger.error(f"Batch search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Data Distribution Analysis ==============

class ValueCount(BaseModel):
    """Count and percentage for a single value"""
    value: str
    count: int
    percentage: float  # 0-100


class FieldDistribution(BaseModel):
    """Distribution statistics for a single field"""
    field_name: str
    field_type: str  # "string", "number", "boolean", "array", "object", "null", "mixed"
    total_count: int
    null_count: int
    unique_count: int

    # Value distribution (for categorical fields)
    value_counts: Optional[List[ValueCount]] = None  # All values with count and percentage

    # For string fields
    length_stats: Optional[Dict[str, Any]] = None  # min, max, avg, median
    length_histogram: Optional[List[Dict[str, Any]]] = None  # [{range, count}]
    top_values: Optional[List[Dict[str, Any]]] = None  # [{value, count}] - deprecated, use value_counts

    # For numeric fields
    numeric_stats: Optional[Dict[str, Any]] = None  # min, max, avg, std, median
    numeric_histogram: Optional[List[Dict[str, Any]]] = None  # [{range, count}]

    # For array fields
    array_length_stats: Optional[Dict[str, Any]] = None


class AnalyzeRequest(BaseModel):
    """Request for data distribution analysis"""
    fields: List[str] = Field(..., min_length=1, max_length=5, description="Fields to analyze (1-5)")
    sample_size: int = Field(10000, ge=100, le=100000, description="Number of records to sample")
    histogram_bins: int = Field(20, ge=5, le=100, description="Number of histogram bins")


class AnalyzeResponse(BaseModel):
    """Response for data distribution analysis"""
    success: bool
    total_records: int
    sampled_records: int
    fields: List[FieldDistribution]
    cross_field_stats: Optional[Dict[str, Any]] = None  # For 2-field correlation


@router.post("/analyze/upload")
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    fields: str = Query(..., description="Comma-separated field names to analyze"),
    sample_size: int = Query(10000, ge=100, le=100000),
    histogram_bins: int = Query(20, ge=5, le=100),
) -> AnalyzeResponse:
    """
    Upload a JSONL file and analyze field distributions.

    Returns distribution statistics for the specified fields including:
    - Value counts and unique values
    - Length distribution (for strings)
    - Numeric statistics (for numbers)
    - Histograms for visualization
    """
    # Validate file type
    if not file.filename.endswith(('.jsonl', '.json', '.ndjson')):
        raise HTTPException(
            status_code=400,
            detail="Only .jsonl, .json, or .ndjson files are supported"
        )

    field_list = [f.strip() for f in fields.split(',') if f.strip()]
    if not field_list:
        raise HTTPException(status_code=400, detail="At least one field is required")
    if len(field_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 fields allowed")

    try:
        content = await file.read()
        lines = content.decode('utf-8').strip().split('\n')

        return _analyze_jsonl_data(
            lines=lines,
            fields=field_list,
            sample_size=sample_size,
            histogram_bins=histogram_bins,
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/path")
async def analyze_file_path(
    file_path: str = Query(..., description="Path to JSONL file"),
    fields: str = Query(..., description="Comma-separated field names to analyze"),
    sample_size: int = Query(10000, ge=100, le=100000),
    histogram_bins: int = Query(20, ge=5, le=100),
) -> AnalyzeResponse:
    """
    Analyze a JSONL file from a local path.

    Use this for large files that are already on the server.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    field_list = [f.strip() for f in fields.split(',') if f.strip()]
    if not field_list:
        raise HTTPException(status_code=400, detail="At least one field is required")
    if len(field_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 fields allowed")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        return _analyze_jsonl_data(
            lines=lines,
            fields=field_list,
            sample_size=sample_size,
            histogram_bins=histogram_bins,
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _analyze_jsonl_data(
    lines: List[str],
    fields: List[str],
    sample_size: int,
    histogram_bins: int,
) -> AnalyzeResponse:
    """
    Core analysis logic for JSONL data.
    """
    import random
    import statistics

    # Parse records
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    total_records = len(records)
    if total_records == 0:
        raise HTTPException(status_code=400, detail="No valid records found")

    # Sample if needed
    if total_records > sample_size:
        records = random.sample(records, sample_size)

    sampled_records = len(records)

    # Analyze each field
    field_distributions = []
    field_values = {}  # Store for cross-field analysis

    for field_name in fields:
        values = []
        null_count = 0

        for record in records:
            value = _get_nested_value(record, field_name)
            if value is None:
                null_count += 1
            values.append(value)

        field_values[field_name] = values
        distribution = _compute_field_distribution(
            field_name=field_name,
            values=values,
            null_count=null_count,
            histogram_bins=histogram_bins,
        )
        field_distributions.append(distribution)

    # Cross-field analysis for 2 fields
    cross_field_stats = None
    if len(fields) == 2:
        cross_field_stats = _compute_cross_field_stats(
            field_values[fields[0]],
            field_values[fields[1]],
            fields[0],
            fields[1],
        )

    return AnalyzeResponse(
        success=True,
        total_records=total_records,
        sampled_records=sampled_records,
        fields=field_distributions,
        cross_field_stats=cross_field_stats,
    )


def _get_nested_value(record: Dict, field_path: str) -> Any:
    """
    Get a value from a nested dictionary using dot notation.
    e.g., "messages.0.content" or "metadata.source"
    """
    parts = field_path.split('.')
    value = record

    for part in parts:
        if value is None:
            return None

        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list):
            try:
                index = int(part)
                value = value[index] if 0 <= index < len(value) else None
            except ValueError:
                return None
        else:
            return None

    return value


def _compute_field_distribution(
    field_name: str,
    values: List[Any],
    null_count: int,
    histogram_bins: int,
) -> FieldDistribution:
    """
    Compute distribution statistics for a field.
    """
    import statistics
    from collections import Counter

    non_null_values = [v for v in values if v is not None]
    total_count = len(values)
    non_null_count = len(non_null_values)

    # Determine field type
    types = set()
    for v in non_null_values[:1000]:  # Sample for type detection
        if isinstance(v, bool):
            types.add("boolean")
        elif isinstance(v, (int, float)):
            types.add("number")
        elif isinstance(v, str):
            types.add("string")
        elif isinstance(v, list):
            types.add("array")
        elif isinstance(v, dict):
            types.add("object")

    if len(types) == 0:
        field_type = "null"
    elif len(types) == 1:
        field_type = types.pop()
    else:
        field_type = "mixed"

    # Compute unique count and value distribution
    try:
        # Convert to hashable for counting
        hashable_values = []
        for v in non_null_values:
            if isinstance(v, (dict, list)):
                hashable_values.append(str(v))
            else:
                hashable_values.append(v)

        unique_count = len(set(hashable_values))

        # Compute value counts with percentages
        value_counter = Counter(hashable_values)
        value_counts = []
        for value, count in value_counter.most_common():
            percentage = round((count / non_null_count * 100) if non_null_count > 0 else 0, 2)
            # Convert value to string for display
            value_str = str(value) if not isinstance(value, str) else value
            # Truncate long values
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            value_counts.append(ValueCount(
                value=value_str,
                count=count,
                percentage=percentage,
            ))

    except Exception as e:
        unique_count = -1
        value_counts = None

    distribution = FieldDistribution(
        field_name=field_name,
        field_type=field_type,
        total_count=total_count,
        null_count=null_count,
        unique_count=unique_count,
        value_counts=value_counts,
    )

    # Type-specific statistics
    if field_type == "string":
        string_values = [v for v in non_null_values if isinstance(v, str)]
        if string_values:
            lengths = [len(v) for v in string_values]
            distribution.length_stats = {
                "min": min(lengths),
                "max": max(lengths),
                "avg": round(statistics.mean(lengths), 2),
                "median": statistics.median(lengths),
            }
            distribution.length_histogram = _compute_histogram(lengths, histogram_bins)
            # Keep top_values for backwards compatibility
            distribution.top_values = _compute_top_values(string_values, top_n=20)

    elif field_type == "number":
        numeric_values = [v for v in non_null_values if isinstance(v, (int, float))]
        if numeric_values:
            distribution.numeric_stats = {
                "min": min(numeric_values),
                "max": max(numeric_values),
                "avg": round(statistics.mean(numeric_values), 4),
                "median": statistics.median(numeric_values),
                "std": round(statistics.stdev(numeric_values), 4) if len(numeric_values) > 1 else 0,
            }
            distribution.numeric_histogram = _compute_histogram(numeric_values, histogram_bins)

    elif field_type == "array":
        array_values = [v for v in non_null_values if isinstance(v, list)]
        if array_values:
            lengths = [len(v) for v in array_values]
            distribution.array_length_stats = {
                "min": min(lengths),
                "max": max(lengths),
                "avg": round(statistics.mean(lengths), 2),
                "median": statistics.median(lengths),
            }

    elif field_type == "boolean":
        bool_values = [v for v in non_null_values if isinstance(v, bool)]
        if bool_values:
            true_count = sum(1 for v in bool_values if v)
            false_count = len(bool_values) - true_count
            total_bool = len(bool_values)
            distribution.top_values = [
                {"value": "true", "count": true_count},
                {"value": "false", "count": false_count},
            ]

    return distribution


def _compute_histogram(values: List[float], bins: int) -> List[Dict[str, Any]]:
    """
    Compute histogram for numeric values.
    """
    if not values:
        return []

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [{"range": f"{min_val}", "min": min_val, "max": max_val, "count": len(values)}]

    bin_width = (max_val - min_val) / bins
    histogram = []

    for i in range(bins):
        bin_min = min_val + i * bin_width
        bin_max = min_val + (i + 1) * bin_width

        if i == bins - 1:
            # Include the max value in the last bin
            count = sum(1 for v in values if bin_min <= v <= bin_max)
        else:
            count = sum(1 for v in values if bin_min <= v < bin_max)

        histogram.append({
            "range": f"{bin_min:.2f}-{bin_max:.2f}",
            "min": round(bin_min, 2),
            "max": round(bin_max, 2),
            "count": count,
        })

    return histogram


def _compute_top_values(values: List[str], top_n: int = 20) -> List[Dict[str, Any]]:
    """
    Compute top N most frequent values.
    """
    from collections import Counter

    # Truncate long strings for display
    truncated = [v[:100] if len(v) > 100 else v for v in values]
    counts = Counter(truncated)

    return [
        {"value": value, "count": count}
        for value, count in counts.most_common(top_n)
    ]


def _compute_cross_field_stats(
    values1: List[Any],
    values2: List[Any],
    name1: str,
    name2: str,
) -> Dict[str, Any]:
    """
    Compute cross-field statistics for correlation analysis.
    """
    # Count co-occurrence patterns
    pairs = list(zip(values1, values2))
    valid_pairs = [(v1, v2) for v1, v2 in pairs if v1 is not None and v2 is not None]

    if not valid_pairs:
        return {"correlation": "no_valid_pairs"}

    # Check if both are numeric
    v1_numeric = all(isinstance(v[0], (int, float)) for v in valid_pairs[:100])
    v2_numeric = all(isinstance(v[1], (int, float)) for v in valid_pairs[:100])

    result = {
        "field1": name1,
        "field2": name2,
        "valid_pairs": len(valid_pairs),
    }

    if v1_numeric and v2_numeric:
        # Compute Pearson correlation coefficient
        import statistics
        import math
        x = [float(v[0]) for v in valid_pairs]
        y = [float(v[1]) for v in valid_pairs]

        if len(x) > 2:
            n = len(x)
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)

            # Calculate covariance and standard deviations
            sum_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            sum_x2 = sum((xi - mean_x) ** 2 for xi in x)
            sum_y2 = sum((yi - mean_y) ** 2 for yi in y)

            if sum_x2 > 0 and sum_y2 > 0:
                correlation = sum_xy / math.sqrt(sum_x2 * sum_y2)
                result["correlation"] = round(correlation, 4)
                result["type"] = "numeric"
            else:
                result["correlation"] = 0
                result["type"] = "constant"
    else:
        # Compute co-occurrence matrix for categorical
        from collections import Counter
        pair_counts = Counter((str(v1)[:50], str(v2)[:50]) for v1, v2 in valid_pairs)
        top_pairs = pair_counts.most_common(20)
        result["top_co_occurrences"] = [
            {"value1": p[0], "value2": p[1], "count": c}
            for (p, c) in top_pairs
        ]
        result["type"] = "categorical"

    return result


@router.post("/analyze/fields")
async def get_available_fields(
    file: UploadFile = File(...),
    sample_lines: int = Query(10, ge=1, le=100, description="Lines to sample for field detection"),
) -> Dict[str, Any]:
    """
    Upload a file and get available fields for analysis.

    Returns a list of field paths found in the first N lines.
    """
    if not file.filename.endswith(('.jsonl', '.json', '.ndjson')):
        raise HTTPException(
            status_code=400,
            detail="Only .jsonl, .json, or .ndjson files are supported"
        )

    try:
        content = await file.read()
        lines = content.decode('utf-8').strip().split('\n')[:sample_lines]

        all_fields = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                fields = _extract_field_paths(record)
                all_fields.update(fields)
            except json.JSONDecodeError:
                continue

        return {
            "success": True,
            "fields": sorted(list(all_fields)),
            "sample_size": len(lines),
        }

    except Exception as e:
        logger.error(f"Field extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_field_paths(obj: Any, prefix: str = "") -> List[str]:
    """
    Extract all field paths from a nested object.
    """
    paths = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.append(path)

            if isinstance(value, dict):
                paths.extend(_extract_field_paths(value, path))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                paths.extend(_extract_field_paths(value[0], f"{path}.0"))

    return paths
