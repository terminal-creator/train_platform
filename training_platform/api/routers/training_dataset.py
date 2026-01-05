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
)

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


# ============== Helper Functions ==============

def detect_label_fields(records: List[Dict], reserved_fields: set = None) -> List[str]:
    """
    Detect fields suitable as labels (string type + low cardinality).

    A field is considered a good label if:
    - It's a string type
    - Cardinality (unique values) is less than 30% of total records
    - Not in reserved fields list
    """
    if reserved_fields is None:
        reserved_fields = {
            'prompt', 'response', 'messages', 'input', 'output',
            'all_model_outputs', 'evaluation', 'reference_script',
            'history_dialogue', 'last_message', 'id', 'uuid'
        }

    if not records:
        return []

    sample_size = min(len(records), 100)
    sample_records = records[:sample_size]

    detected = []
    for field in records[0].keys():
        if field in reserved_fields:
            continue

        # Get values for this field
        values = []
        for r in sample_records:
            val = r.get(field)
            if val is not None and isinstance(val, str):
                values.append(val)

        # Check if it's a good label field
        if len(values) >= sample_size * 0.5:  # At least 50% non-null
            unique_ratio = len(set(values)) / len(values) if values else 1
            if unique_ratio < 0.3:  # Less than 30% unique (good for grouping)
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
            prompt_field=prompt_field,
            response_field=response_field,
            analyzed_at=datetime.utcnow(),
        )

        repo = TrainingDatasetRepository(session)
        dataset = repo.create(dataset)

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
            created_at=dataset.created_at,
            analyzed_at=dataset.analyzed_at,
        )

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

    return [
        TrainingDatasetResponse(
            uuid=d.uuid,
            name=d.name,
            description=d.description,
            file_path=d.file_path,
            file_format=d.file_format,
            file_size_mb=d.file_size_mb,
            total_rows=d.total_rows,
            columns=d.columns,
            label_fields=d.label_fields,
            field_distributions=d.field_distributions,
            prompt_field=d.prompt_field,
            response_field=d.response_field,
            created_at=d.created_at,
            analyzed_at=d.analyzed_at,
        )
        for d in datasets
    ]


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
        created_at=dataset.created_at,
        analyzed_at=dataset.analyzed_at,
    )


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

    # Delete file
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)

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

    # Compute loss segments
    loss_segments = []

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

        # Compute loss segments
        loss_segments = []
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
        created_at=dataset.created_at,
        analyzed_at=dataset.analyzed_at,
    )


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
        created_at=dataset.created_at,
        analyzed_at=dataset.analyzed_at,
    )


@router.post("/{uuid}/reanalyze", response_model=TrainingDatasetResponse)
async def reanalyze_dataset(
    uuid: str,
    auto_detect_labels: bool = Query(False, description="Re-detect label fields automatically"),
    session: Session = Depends(get_session),
):
    """Reanalyze a dataset to update distributions."""
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

    # Recompute distributions
    dataset.field_distributions = compute_field_distributions(records, dataset.label_fields)
    dataset.analyzed_at = datetime.utcnow()

    dataset = repo.update(dataset)

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
        created_at=dataset.created_at,
        analyzed_at=dataset.analyzed_at,
    )
