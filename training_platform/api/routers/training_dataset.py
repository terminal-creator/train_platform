"""
Training Dataset API Router

Provides endpoints for training dataset management:
- Upload training datasets (JSON, JSONL, Parquet)
- Analyze label field distributions
- View individual samples with loss highlighting
- Configure label and loss computation fields
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlmodel import Session
import logging
import os
import uuid
import json
import tempfile
import shutil

from ...core.database import (
    get_session,
    TrainingDataset,
    TrainingDatasetRepository,
    DatasetSyncStatus,
)
from ...core.ssh_runner import get_dataset_sync_service
from ...core.run_mode import load_run_mode_config, RunMode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training-datasets", tags=["Training Datasets"])

# Constants
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "training_datasets")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============== Request/Response Models ==============

class TrainingDatasetResponse(BaseModel):
    """Training dataset response"""
    uuid: str
    name: str
    description: Optional[str] = None
    file_path: str
    file_format: str
    file_size_mb: float
    total_rows: int
    columns: List[str]
    label_fields: List[str]
    field_distributions: Dict[str, Any]
    prompt_field: str
    response_field: str
    # Remote sync info
    remote_path: Optional[str] = None
    sync_status: str = "not_synced"
    sync_error: Optional[str] = None
    synced_at: Optional[datetime] = None
    # Timestamps
    created_at: datetime
    analyzed_at: Optional[datetime] = None


class UploadTrainingDatasetRequest(BaseModel):
    """Request parameters for dataset upload"""
    name: str
    description: Optional[str] = None
    label_fields: Optional[List[str]] = None  # Auto-detect if not provided
    prompt_field: str = "prompt"
    response_field: str = "response"


class ConfigureLabelFieldsRequest(BaseModel):
    """Request to configure label fields"""
    label_fields: List[str]


class ConfigureLossFieldsRequest(BaseModel):
    """Request to configure loss computation fields"""
    prompt_field: str
    response_field: str


class LossSegment(BaseModel):
    """A segment of text with loss computation info"""
    field: str
    text: str
    computes_loss: bool


class SampleResponse(BaseModel):
    """Response for a single sample with loss highlighting"""
    index: int
    data: Dict[str, Any]
    loss_segments: List[LossSegment]
    labels: Dict[str, Any]  # {field_name: value}


class DistributionResponse(BaseModel):
    """Response for field distribution"""
    field: str
    distribution: Dict[str, int]
    total: int


class DatasetStatsResponse(BaseModel):
    """Response for dataset statistics overview"""
    total_samples: int
    avg_turns: float
    avg_prompt_chars: float
    avg_response_chars: float
    avg_total_chars: float
    format_type: str  # 'messages' or 'prompt_response'
    has_system_prompt: float  # percentage
    # Length distribution buckets
    prompt_length_distribution: Dict[str, int]
    response_length_distribution: Dict[str, int]
    turns_distribution: Dict[str, int]


class QualityIssue(BaseModel):
    """A quality issue found in the dataset"""
    issue_type: str
    count: int
    percentage: float
    sample_indices: List[int]  # First few indices with this issue


class QualityCheckResponse(BaseModel):
    """Response for quality check"""
    total_samples: int
    issues_found: int
    quality_score: float  # 0-100
    issues: List[QualityIssue]


# ============== Helper Functions ==============

def _get_length_bucket(length: int) -> str:
    """Get length bucket label for distribution."""
    if length < 50:
        return "0-50"
    elif length < 100:
        return "50-100"
    elif length < 200:
        return "100-200"
    elif length < 500:
        return "200-500"
    elif length < 1000:
        return "500-1k"
    elif length < 2000:
        return "1k-2k"
    else:
        return "2k+"


def _get_turns_bucket(turns: int) -> str:
    """Get turns bucket label."""
    if turns == 1:
        return "1轮"
    elif turns == 2:
        return "2轮"
    elif turns == 3:
        return "3轮"
    elif turns <= 5:
        return "4-5轮"
    else:
        return "6+轮"


def _analyze_record(record: Dict) -> Dict:
    """Analyze a single record for statistics."""
    result = {
        "format": "unknown",
        "turns": 0,
        "prompt_chars": 0,
        "response_chars": 0,
        "total_chars": 0,
        "has_system": False,
    }

    if "messages" in record and isinstance(record["messages"], list):
        result["format"] = "messages"
        messages = record["messages"]

        prompt_chars = 0
        response_chars = 0
        turns = 0

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            if role == "system":
                result["has_system"] = True
                prompt_chars += len(content)
            elif role == "user":
                prompt_chars += len(content)
                turns += 1
            elif role == "assistant":
                response_chars += len(content)

        result["turns"] = max(turns, 1)
        result["prompt_chars"] = prompt_chars
        result["response_chars"] = response_chars
        result["total_chars"] = prompt_chars + response_chars
    else:
        result["format"] = "prompt_response"
        result["turns"] = 1

        prompt = record.get("prompt", "")
        response = record.get("response", "")

        result["prompt_chars"] = len(str(prompt))
        result["response_chars"] = len(str(response))
        result["total_chars"] = result["prompt_chars"] + result["response_chars"]

    return result


def _check_quality_issues(records: List[Dict]) -> List[QualityIssue]:
    """Check for quality issues in the dataset."""
    issues = []

    # Track issues
    short_response_indices = []
    empty_response_indices = []
    very_long_indices = []
    duplicate_prompts = {}

    for i, record in enumerate(records):
        analysis = _analyze_record(record)

        # Check for empty response
        if analysis["response_chars"] == 0:
            empty_response_indices.append(i)
        # Check for short response (< 20 chars)
        elif analysis["response_chars"] < 20:
            short_response_indices.append(i)

        # Check for very long samples (> 4000 chars total)
        if analysis["total_chars"] > 4000:
            very_long_indices.append(i)

        # Check for duplicate prompts (simplified check)
        if "messages" in record:
            # Get user messages as key
            user_msgs = [m.get("content", "")[:100] for m in record.get("messages", []) if m.get("role") == "user"]
            prompt_key = "|".join(user_msgs)
        else:
            prompt_key = str(record.get("prompt", ""))[:100]

        if prompt_key:
            if prompt_key not in duplicate_prompts:
                duplicate_prompts[prompt_key] = []
            duplicate_prompts[prompt_key].append(i)

    total = len(records)

    # Empty responses
    if empty_response_indices:
        issues.append(QualityIssue(
            issue_type="空回复",
            count=len(empty_response_indices),
            percentage=round(len(empty_response_indices) / total * 100, 1),
            sample_indices=empty_response_indices[:5],
        ))

    # Short responses
    if short_response_indices:
        issues.append(QualityIssue(
            issue_type="回复过短(<20字)",
            count=len(short_response_indices),
            percentage=round(len(short_response_indices) / total * 100, 1),
            sample_indices=short_response_indices[:5],
        ))

    # Very long samples
    if very_long_indices:
        issues.append(QualityIssue(
            issue_type="内容过长(>4000字)",
            count=len(very_long_indices),
            percentage=round(len(very_long_indices) / total * 100, 1),
            sample_indices=very_long_indices[:5],
        ))

    # Duplicates
    duplicate_indices = []
    for key, indices in duplicate_prompts.items():
        if len(indices) > 1:
            duplicate_indices.extend(indices[1:])  # Keep first, mark rest as duplicates

    if duplicate_indices:
        issues.append(QualityIssue(
            issue_type="疑似重复",
            count=len(duplicate_indices),
            percentage=round(len(duplicate_indices) / total * 100, 1),
            sample_indices=duplicate_indices[:5],
        ))

    return issues


def compute_dataset_statistics(records: List[Dict]) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a dataset.
    Returns a dict that can be stored in the statistics JSON field.
    """
    if not records:
        return {}

    # Analyze all records
    analyses = [_analyze_record(r) for r in records]
    total = len(analyses)

    # Calculate averages
    avg_turns = sum(a["turns"] for a in analyses) / total
    avg_prompt_chars = sum(a["prompt_chars"] for a in analyses) / total
    avg_response_chars = sum(a["response_chars"] for a in analyses) / total
    avg_total_chars = sum(a["total_chars"] for a in analyses) / total

    # Determine format type
    format_counts = {}
    for a in analyses:
        fmt = a["format"]
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    format_type = max(format_counts, key=format_counts.get)

    # System prompt percentage
    has_system_count = sum(1 for a in analyses if a["has_system"])
    has_system_pct = round(has_system_count / total * 100, 1)

    # Length distributions
    prompt_length_dist = {}
    response_length_dist = {}
    turns_dist = {}

    for a in analyses:
        # Prompt length bucket
        bucket = _get_length_bucket(a["prompt_chars"])
        prompt_length_dist[bucket] = prompt_length_dist.get(bucket, 0) + 1

        # Response length bucket
        bucket = _get_length_bucket(a["response_chars"])
        response_length_dist[bucket] = response_length_dist.get(bucket, 0) + 1

        # Turns bucket
        bucket = _get_turns_bucket(a["turns"])
        turns_dist[bucket] = turns_dist.get(bucket, 0) + 1

    return {
        "total_samples": total,
        "avg_turns": round(avg_turns, 2),
        "avg_prompt_chars": round(avg_prompt_chars, 0),
        "avg_response_chars": round(avg_response_chars, 0),
        "avg_total_chars": round(avg_total_chars, 0),
        "format_type": format_type,
        "has_system_prompt": has_system_pct,
        "prompt_length_distribution": prompt_length_dist,
        "response_length_distribution": response_length_dist,
        "turns_distribution": turns_dist,
    }


def compute_quality_statistics(records: List[Dict]) -> Dict[str, Any]:
    """
    Compute quality statistics for a dataset.
    Returns a dict that can be stored in the quality_stats JSON field.
    """
    if not records:
        return {"quality_score": 100.0, "issues_found": 0, "issues": []}

    # Check for issues
    issues = _check_quality_issues(records)

    # Calculate quality score (100 - penalty for each issue type)
    total_issues = sum(issue.count for issue in issues)
    issue_ratio = total_issues / len(records)
    quality_score = max(0, 100 - issue_ratio * 100)

    # Convert issues to dict format for JSON storage
    issues_list = [
        {
            "issue_type": issue.issue_type,
            "count": issue.count,
            "percentage": issue.percentage,
            "sample_indices": issue.sample_indices,
        }
        for issue in issues
    ]

    return {
        "total_samples": len(records),
        "quality_score": round(quality_score, 1),
        "issues_found": total_issues,
        "issues": issues_list,
    }


def detect_label_fields(records: List[Dict], reserved_fields: set = None) -> List[str]:
    """
    Detect fields suitable as labels (categorical data with low cardinality).

    A field is considered a good label if:
    - It's a simple type (not dict/list)
    - Short average length (< 100 chars) - not long text content
    - Low cardinality: either < 30 unique values OR < 30% unique ratio
    - At least 50% non-null values
    """
    # Only exclude truly special fields, detect everything else dynamically
    if reserved_fields is None:
        reserved_fields = {'id', 'uuid', '_id'}

    if not records:
        return []

    sample_size = min(len(records), 200)  # Use larger sample for better detection
    sample_records = records[:sample_size]

    detected = []
    for field in records[0].keys():
        if field in reserved_fields:
            continue

        # Get values for this field
        values = []
        total_length = 0
        for r in sample_records:
            val = r.get(field)
            # Skip complex types (dict, list)
            if isinstance(val, (dict, list)):
                break
            if val is not None:
                str_val = str(val)
                values.append(str_val)
                total_length += len(str_val)
        else:
            # Only proceed if we didn't break (no complex types found)
            if len(values) >= sample_size * 0.5:  # At least 50% non-null
                avg_length = total_length / len(values) if values else 0
                unique_count = len(set(values))
                unique_ratio = unique_count / len(values) if values else 1

                # Good label criteria:
                # 1. Short values (not long text content)
                # 2. Low cardinality: < 30 unique values OR < 30% unique ratio
                # This catches small datasets with few categories AND large datasets
                if avg_length < 100 and (unique_count < 30 or unique_ratio < 0.3):
                    detected.append(field)

    return detected


def compute_field_distributions(records: List[Dict], label_fields: List[str]) -> Dict[str, Dict[str, int]]:
    """Compute value distribution for each label field."""
    distributions = {}

    for field in label_fields:
        counter = {}
        for record in records:
            value = record.get(field)
            if value is not None:
                str_value = str(value)
                counter[str_value] = counter.get(str_value, 0) + 1
        distributions[field] = counter

    return distributions


def load_dataset_records(file_path: str, file_format: str) -> List[Dict]:
    """Load records from a dataset file."""
    records = []

    if file_format == "jsonl":
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    elif file_format == "json":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                records = [data]
    elif file_format == "parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            records = df.to_dict('records')
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="Parquet support requires pandas. Install with: pip install pandas pyarrow"
            )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {file_format}")

    return records


def _build_dataset_response(dataset: TrainingDataset) -> TrainingDatasetResponse:
    """Build TrainingDatasetResponse from TrainingDataset model."""
    return TrainingDatasetResponse(
        uuid=dataset.uuid,
        name=dataset.name,
        description=dataset.description,
        file_path=dataset.file_path,
        file_format=dataset.file_format,
        file_size_mb=dataset.file_size_mb,
        total_rows=dataset.total_rows,
        columns=dataset.columns,
        label_fields=dataset.label_fields,
        field_distributions=dataset.field_distributions,
        prompt_field=dataset.prompt_field,
        response_field=dataset.response_field,
        remote_path=dataset.remote_path,
        sync_status=dataset.sync_status.value if dataset.sync_status else "not_synced",
        sync_error=dataset.sync_error,
        synced_at=dataset.synced_at,
        created_at=dataset.created_at,
        analyzed_at=dataset.analyzed_at,
    )


def _sync_dataset_to_remote(dataset: TrainingDataset, repo: TrainingDatasetRepository) -> TrainingDataset:
    """
    Sync dataset to remote server if SSH mode is configured.
    Updates dataset with sync status.
    """
    config = load_run_mode_config()
    if config.mode != RunMode.SSH:
        # Not in SSH mode, skip sync
        return dataset

    sync_service = get_dataset_sync_service()
    if not sync_service:
        dataset.sync_status = DatasetSyncStatus.FAILED
        dataset.sync_error = "SSH not configured properly"
        return repo.update(dataset)

    # Update status to syncing
    dataset.sync_status = DatasetSyncStatus.SYNCING
    dataset.sync_error = None
    repo.update(dataset)

    try:
        result = sync_service.sync_dataset(
            local_path=dataset.file_path,
            dataset_name=dataset.name,
            dataset_uuid=dataset.uuid,
        )

        if result.get("success"):
            dataset.remote_path = result.get("remote_path")
            dataset.sync_status = DatasetSyncStatus.SYNCED
            dataset.sync_error = None
            dataset.synced_at = datetime.utcnow()
            logger.info(f"Dataset {dataset.uuid} synced to {dataset.remote_path}")
        else:
            dataset.sync_status = DatasetSyncStatus.FAILED
            dataset.sync_error = result.get("error", "Unknown error")
            logger.error(f"Dataset {dataset.uuid} sync failed: {dataset.sync_error}")

    except Exception as e:
        dataset.sync_status = DatasetSyncStatus.FAILED
        dataset.sync_error = str(e)
        logger.error(f"Dataset {dataset.uuid} sync exception: {e}")
    finally:
        sync_service.disconnect()

    return repo.update(dataset)


# ============== API Endpoints ==============

@router.post("/upload", response_model=TrainingDatasetResponse)
async def upload_training_dataset(
    file: UploadFile = File(...),
    name: str = Query(..., description="Dataset name"),
    description: Optional[str] = Query(None, description="Dataset description"),
    label_fields: Optional[str] = Query(None, description="Comma-separated label fields (auto-detect if empty)"),
    prompt_field: str = Query("prompt", description="Field name for prompt (for loss computation)"),
    response_field: str = Query("response", description="Field name for response (for loss computation)"),
    session: Session = Depends(get_session),
):
    """
    Upload a training dataset.

    Supports JSON, JSONL, and Parquet formats.
    Label fields will be auto-detected if not specified.
    """
    # Determine file format
    filename = file.filename.lower()
    if filename.endswith('.jsonl') or filename.endswith('.ndjson'):
        file_format = "jsonl"
    elif filename.endswith('.json'):
        file_format = "json"
    elif filename.endswith('.parquet'):
        file_format = "parquet"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Use .json, .jsonl, .ndjson, or .parquet"
        )

    # Generate UUID and save file
    dataset_uuid = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{dataset_uuid}.{file_format}")

    try:
        # Save uploaded file
        content = await file.read()
        with open(save_path, 'wb') as f:
            f.write(content)

        # Calculate file size
        file_size_mb = len(content) / (1024 * 1024)

        # Load and analyze records
        records = load_dataset_records(save_path, file_format)
        if not records:
            os.remove(save_path)
            raise HTTPException(status_code=400, detail="No valid records found in file")

        # Get columns
        columns = list(records[0].keys()) if records else []

        # Parse or detect label fields
        if label_fields:
            parsed_label_fields = [f.strip() for f in label_fields.split(',') if f.strip()]
        else:
            parsed_label_fields = detect_label_fields(records)

        # Compute distributions
        field_distributions = compute_field_distributions(records, parsed_label_fields)

        # Pre-compute statistics and quality (one-time on upload)
        statistics = compute_dataset_statistics(records)
        quality_stats = compute_quality_statistics(records)

        # Create database record
        dataset = TrainingDataset(
            uuid=dataset_uuid,
            name=name,
            description=description,
            file_path=save_path,
            file_format=file_format,
            file_size_mb=round(file_size_mb, 2),
            total_rows=len(records),
            columns=columns,
            label_fields=parsed_label_fields,
            field_distributions=field_distributions,
            statistics=statistics,
            quality_stats=quality_stats,
            prompt_field=prompt_field,
            response_field=response_field,
            analyzed_at=datetime.utcnow(),
        )

        repo = TrainingDatasetRepository(session)
        dataset = repo.create(dataset)

        # Auto-sync to remote server if SSH mode is configured
        dataset = _sync_dataset_to_remote(dataset, repo)

        return _build_dataset_response(dataset)

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if os.path.exists(save_path):
            os.remove(save_path)
        logger.error(f"Failed to upload dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[TrainingDatasetResponse])
async def list_training_datasets(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
):
    """List all training datasets."""
    repo = TrainingDatasetRepository(session)
    datasets, total = repo.list_datasets(offset=offset, limit=limit)

    return [_build_dataset_response(d) for d in datasets]


@router.get("/{uuid}", response_model=TrainingDatasetResponse)
async def get_training_dataset(
    uuid: str,
    session: Session = Depends(get_session),
):
    """Get a training dataset by UUID."""
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return _build_dataset_response(dataset)


@router.delete("/{uuid}")
async def delete_training_dataset(
    uuid: str,
    session: Session = Depends(get_session),
):
    """Delete a training dataset."""
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete local file
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)

    # Delete remote file if synced
    if dataset.remote_path and dataset.sync_status == DatasetSyncStatus.SYNCED:
        try:
            sync_service = get_dataset_sync_service()
            if sync_service:
                sync_service.delete_remote_dataset(dataset.remote_path)
                sync_service.disconnect()
        except Exception as e:
            logger.warning(f"Failed to delete remote file: {e}")

    # Delete from database
    repo.delete(uuid)

    return {"success": True, "message": "Dataset deleted"}


@router.get("/{uuid}/distribution", response_model=List[DistributionResponse])
async def get_distribution(
    uuid: str,
    fields: Optional[str] = Query(None, description="Comma-separated field names (defaults to all label_fields)"),
    session: Session = Depends(get_session),
):
    """Get distribution for specified label fields."""
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Parse fields
    if fields:
        field_list = [f.strip() for f in fields.split(',') if f.strip()]
    else:
        field_list = dataset.label_fields

    # Return distributions
    results = []
    for field in field_list:
        if field in dataset.field_distributions:
            dist = dataset.field_distributions[field]
            results.append(DistributionResponse(
                field=field,
                distribution=dist,
                total=sum(dist.values()),
            ))

    return results


@router.get("/{uuid}/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(
    uuid: str,
    recompute: bool = Query(False, description="Force recompute statistics from file"),
    session: Session = Depends(get_session),
):
    """
    Get comprehensive statistics for a dataset.

    Returns cached statistics by default (computed on upload).
    Use recompute=true to force recalculation from file.
    """
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Return cached stats if available and not forcing recompute
    if dataset.statistics and not recompute:
        stats = dataset.statistics
        return DatasetStatsResponse(
            total_samples=stats.get("total_samples", dataset.total_rows),
            avg_turns=stats.get("avg_turns", 1),
            avg_prompt_chars=stats.get("avg_prompt_chars", 0),
            avg_response_chars=stats.get("avg_response_chars", 0),
            avg_total_chars=stats.get("avg_total_chars", 0),
            format_type=stats.get("format_type", "unknown"),
            has_system_prompt=stats.get("has_system_prompt", 0),
            prompt_length_distribution=stats.get("prompt_length_distribution", {}),
            response_length_distribution=stats.get("response_length_distribution", {}),
            turns_distribution=stats.get("turns_distribution", {}),
        )

    # Recompute from file (for legacy datasets or forced recompute)
    records = load_dataset_records(dataset.file_path, dataset.file_format)

    if not records:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    stats = compute_dataset_statistics(records)

    # Update cached stats in database
    dataset.statistics = stats
    dataset.analyzed_at = datetime.utcnow()
    repo.update(dataset)

    return DatasetStatsResponse(
        total_samples=stats["total_samples"],
        avg_turns=stats["avg_turns"],
        avg_prompt_chars=stats["avg_prompt_chars"],
        avg_response_chars=stats["avg_response_chars"],
        avg_total_chars=stats["avg_total_chars"],
        format_type=stats["format_type"],
        has_system_prompt=stats["has_system_prompt"],
        prompt_length_distribution=stats["prompt_length_distribution"],
        response_length_distribution=stats["response_length_distribution"],
        turns_distribution=stats["turns_distribution"],
    )


@router.get("/{uuid}/quality-check", response_model=QualityCheckResponse)
async def check_dataset_quality(
    uuid: str,
    recompute: bool = Query(False, description="Force recompute quality check from file"),
    session: Session = Depends(get_session),
):
    """
    Run quality checks on the dataset.

    Returns cached quality stats by default (computed on upload).
    Use recompute=true to force recalculation from file.
    """
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Return cached quality stats if available and not forcing recompute
    if dataset.quality_stats and not recompute:
        qstats = dataset.quality_stats
        # Convert stored issues back to QualityIssue objects
        issues = [
            QualityIssue(
                issue_type=issue["issue_type"],
                count=issue["count"],
                percentage=issue["percentage"],
                sample_indices=issue["sample_indices"],
            )
            for issue in qstats.get("issues", [])
        ]
        return QualityCheckResponse(
            total_samples=qstats.get("total_samples", dataset.total_rows),
            issues_found=qstats.get("issues_found", 0),
            quality_score=qstats.get("quality_score", 100.0),
            issues=issues,
        )

    # Recompute from file (for legacy datasets or forced recompute)
    records = load_dataset_records(dataset.file_path, dataset.file_format)

    if not records:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    qstats = compute_quality_statistics(records)

    # Update cached quality stats in database
    dataset.quality_stats = qstats
    repo.update(dataset)

    # Convert to QualityIssue objects for response
    issues = [
        QualityIssue(
            issue_type=issue["issue_type"],
            count=issue["count"],
            percentage=issue["percentage"],
            sample_indices=issue["sample_indices"],
        )
        for issue in qstats["issues"]
    ]

    return QualityCheckResponse(
        total_samples=qstats["total_samples"],
        issues_found=qstats["issues_found"],
        quality_score=qstats["quality_score"],
        issues=issues,
    )


def _build_loss_segments_from_messages(messages: List[Dict]) -> List[LossSegment]:
    """Build loss segments from OpenAI messages format."""
    segments = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            segments.append(LossSegment(
                field="system",
                text=content,
                computes_loss=False,
            ))
        elif role == "user":
            segments.append(LossSegment(
                field="user",
                text=content,
                computes_loss=False,
            ))
        elif role == "assistant":
            segments.append(LossSegment(
                field="assistant",
                text=content,
                computes_loss=True,  # Only assistant responses compute loss
            ))
    return segments


@router.get("/{uuid}/sample/{index}", response_model=SampleResponse)
async def get_sample(
    uuid: str,
    index: int,
    session: Session = Depends(get_session),
):
    """
    Get a single sample with loss highlighting.

    Returns the sample data along with segments indicating which parts
    are used for loss computation (typically only the response field).

    Supports both formats:
    - OpenAI messages format: {"messages": [{"role": "...", "content": "..."}]}
    - Simple format: {"prompt": "...", "response": "..."}
    """
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if index < 0 or index >= dataset.total_rows:
        raise HTTPException(status_code=400, detail=f"Index out of range (0-{dataset.total_rows - 1})")

    # Load the specific record
    records = load_dataset_records(dataset.file_path, dataset.file_format)
    if index >= len(records):
        raise HTTPException(status_code=400, detail="Index out of range")

    record = records[index]

    # Compute loss segments - check for messages format first
    loss_segments = []

    if "messages" in record and isinstance(record["messages"], list):
        # OpenAI messages format
        loss_segments = _build_loss_segments_from_messages(record["messages"])
    else:
        # Simple prompt/response format
        # Add prompt segment (no loss)
        prompt_text = record.get(dataset.prompt_field, "")
        if prompt_text:
            loss_segments.append(LossSegment(
                field=dataset.prompt_field,
                text=str(prompt_text),
                computes_loss=False,
            ))

        # Add response segment (with loss)
        response_text = record.get(dataset.response_field, "")
        if response_text:
            loss_segments.append(LossSegment(
                field=dataset.response_field,
                text=str(response_text),
                computes_loss=True,
            ))

    # Get label values
    labels = {}
    for field in dataset.label_fields:
        if field in record:
            labels[field] = record[field]

    return SampleResponse(
        index=index,
        data=record,
        loss_segments=loss_segments,
        labels=labels,
    )


@router.get("/{uuid}/samples", response_model=List[SampleResponse])
async def get_samples(
    uuid: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    label_filter: Optional[str] = Query(None, description="Filter by label field:value"),
    session: Session = Depends(get_session),
):
    """
    Get multiple samples with pagination and optional filtering.

    Use label_filter in format "field:value" to filter by label.
    """
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Load records
    records = load_dataset_records(dataset.file_path, dataset.file_format)

    # Apply filter if specified
    if label_filter and ':' in label_filter:
        field, value = label_filter.split(':', 1)
        records = [r for r in records if str(r.get(field)) == value]

    # Apply pagination
    paginated = records[offset:offset + limit]

    results = []
    for i, record in enumerate(paginated):
        actual_index = offset + i

        # Compute loss segments - check for messages format first
        loss_segments = []

        if "messages" in record and isinstance(record["messages"], list):
            # OpenAI messages format - truncate content for list view
            for msg in record["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                truncated = content[:500] + ("..." if len(content) > 500 else "")

                loss_segments.append(LossSegment(
                    field=role,
                    text=truncated,
                    computes_loss=(role == "assistant"),
                ))
        else:
            # Simple prompt/response format
            prompt_text = record.get(dataset.prompt_field, "")
            if prompt_text:
                loss_segments.append(LossSegment(
                    field=dataset.prompt_field,
                    text=str(prompt_text)[:500] + ("..." if len(str(prompt_text)) > 500 else ""),
                    computes_loss=False,
                ))

            response_text = record.get(dataset.response_field, "")
            if response_text:
                loss_segments.append(LossSegment(
                    field=dataset.response_field,
                    text=str(response_text)[:500] + ("..." if len(str(response_text)) > 500 else ""),
                    computes_loss=True,
                ))

        # Get label values
        labels = {}
        for field in dataset.label_fields:
            if field in record:
                labels[field] = record[field]

        results.append(SampleResponse(
            index=actual_index,
            data={k: (str(v)[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
                  for k, v in record.items()},
            loss_segments=loss_segments,
            labels=labels,
        ))

    return results


@router.post("/{uuid}/configure-labels", response_model=TrainingDatasetResponse)
async def configure_label_fields(
    uuid: str,
    request: ConfigureLabelFieldsRequest,
    session: Session = Depends(get_session),
):
    """Configure which fields to use as labels for distribution analysis."""
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate fields exist
    for field in request.label_fields:
        if field not in dataset.columns:
            raise HTTPException(status_code=400, detail=f"Field not found: {field}")

    # Recompute distributions
    records = load_dataset_records(dataset.file_path, dataset.file_format)
    field_distributions = compute_field_distributions(records, request.label_fields)

    # Update dataset
    dataset.label_fields = request.label_fields
    dataset.field_distributions = field_distributions
    dataset.analyzed_at = datetime.utcnow()

    dataset = repo.update(dataset)

    return _build_dataset_response(dataset)


@router.post("/{uuid}/configure-loss", response_model=TrainingDatasetResponse)
async def configure_loss_fields(
    uuid: str,
    request: ConfigureLossFieldsRequest,
    session: Session = Depends(get_session),
):
    """Configure which fields are used for loss computation."""
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate fields exist
    if request.prompt_field not in dataset.columns:
        raise HTTPException(status_code=400, detail=f"Prompt field not found: {request.prompt_field}")
    if request.response_field not in dataset.columns:
        raise HTTPException(status_code=400, detail=f"Response field not found: {request.response_field}")

    # Update dataset
    dataset.prompt_field = request.prompt_field
    dataset.response_field = request.response_field

    dataset = repo.update(dataset)

    return _build_dataset_response(dataset)


@router.post("/{uuid}/reanalyze", response_model=TrainingDatasetResponse)
async def reanalyze_dataset(
    uuid: str,
    auto_detect_labels: bool = Query(False, description="Re-detect label fields automatically"),
    session: Session = Depends(get_session),
):
    """Reanalyze a dataset to update distributions, statistics, and quality."""
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Reload and reanalyze
    records = load_dataset_records(dataset.file_path, dataset.file_format)

    # Update basic stats
    dataset.total_rows = len(records)
    dataset.columns = list(records[0].keys()) if records else []

    # Optionally re-detect labels
    if auto_detect_labels:
        dataset.label_fields = detect_label_fields(records)

    # Recompute all analytics
    dataset.field_distributions = compute_field_distributions(records, dataset.label_fields)
    dataset.statistics = compute_dataset_statistics(records)
    dataset.quality_stats = compute_quality_statistics(records)
    dataset.analyzed_at = datetime.utcnow()

    dataset = repo.update(dataset)

    return _build_dataset_response(dataset)


@router.post("/{uuid}/sync", response_model=TrainingDatasetResponse)
async def sync_dataset_to_remote(
    uuid: str,
    force: bool = Query(False, description="Force re-sync even if already synced"),
    session: Session = Depends(get_session),
):
    """
    Manually sync a dataset to the remote SSH server.

    Use force=true to re-sync even if already synced.
    Requires SSH mode to be configured in Settings.
    """
    repo = TrainingDatasetRepository(session)
    dataset = repo.get_by_uuid(uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check if SSH mode is configured
    config = load_run_mode_config()
    if config.mode != RunMode.SSH:
        raise HTTPException(
            status_code=400,
            detail="SSH mode not configured. Go to Settings to configure SSH connection."
        )

    # Skip if already synced and not forcing
    if dataset.sync_status == DatasetSyncStatus.SYNCED and not force:
        return _build_dataset_response(dataset)

    # Perform sync
    dataset = _sync_dataset_to_remote(dataset, repo)

    return _build_dataset_response(dataset)


@router.post("/sync-all")
async def sync_all_datasets(
    force: bool = Query(False, description="Force re-sync all datasets"),
    session: Session = Depends(get_session),
):
    """
    Sync all datasets to the remote SSH server.

    Useful after configuring SSH mode for the first time.
    """
    config = load_run_mode_config()
    if config.mode != RunMode.SSH:
        raise HTTPException(
            status_code=400,
            detail="SSH mode not configured. Go to Settings to configure SSH connection."
        )

    repo = TrainingDatasetRepository(session)
    datasets, _ = repo.list_datasets(offset=0, limit=1000)

    results = {
        "total": len(datasets),
        "synced": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }

    for dataset in datasets:
        if dataset.sync_status == DatasetSyncStatus.SYNCED and not force:
            results["skipped"] += 1
            continue

        try:
            dataset = _sync_dataset_to_remote(dataset, repo)
            if dataset.sync_status == DatasetSyncStatus.SYNCED:
                results["synced"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "uuid": dataset.uuid,
                    "name": dataset.name,
                    "error": dataset.sync_error,
                })
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "uuid": dataset.uuid,
                "name": dataset.name,
                "error": str(e),
            })

    return results
