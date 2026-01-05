"""
Evaluation API Router

Provides endpoints for:
- Evaluation dataset management (CRUD)
- Triggering evaluations on checkpoints
- Viewing evaluation results
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import logging
import os
import uuid
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlmodel import Session
from ...core.database import (
    get_session,
    engine,
    EvalDataset,
    EvalTask,
    EvalComparison,
    EvalDatasetRepository,
    EvalTaskRepository,
    EvalComparisonRepository,
    CheckpointRepository,
    JobRepository,
    EvalDatasetFormat,
    EvalCapability,
    EvalMethod,
    EvalTaskStatus,
)
from ...core.inference import (
    VLLMInferenceEngine,
    InferenceConfig,
    infer_with_vllm,
    batch_infer_with_vllm,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])

# Directory for evaluation datasets
EVAL_DATASETS_DIR = os.environ.get("EVAL_DATASETS_DIR", "./eval_datasets")


# ============== Pydantic Models ==============

class EvalDatasetResponse(BaseModel):
    """Response for evaluation dataset"""
    uuid: str
    name: str
    description: Optional[str]
    format: str
    capability: str
    eval_method: str
    sample_count: int
    label_field: Optional[str] = None
    available_labels: List[str] = []
    judge_prompt_template: Optional[str] = None
    judge_config: Dict[str, Any]
    created_at: datetime


class EvalDatasetListResponse(BaseModel):
    """Response for listing evaluation datasets"""
    datasets: List[EvalDatasetResponse]
    total: int


class EvalDatasetPreview(BaseModel):
    """Preview of evaluation dataset"""
    uuid: str
    name: str
    format: str
    sample_count: int
    samples: List[Dict[str, Any]]


class ModelType(str, Enum):
    API = "api"                  # API-based inference (OpenAI compatible)
    LOCAL_MODEL = "local_model"  # Local pre-training model
    CHECKPOINT = "checkpoint"    # Post-training checkpoint


class TriggerEvalRequest(BaseModel):
    """Request to trigger evaluation"""
    dataset_uuids: List[str]
    name: Optional[str] = None  # Optional name for the evaluation run (for comparisons)
    model_type: ModelType = ModelType.CHECKPOINT
    # For checkpoint mode
    checkpoint_id: Optional[int] = None
    # For local model mode
    model_path: Optional[str] = None
    # For API mode
    api_base_url: Optional[str] = None
    api_model: Optional[str] = None
    api_key_env: Optional[str] = "DASHSCOPE_API_KEY"


class EvalTaskResponse(BaseModel):
    """Response for evaluation task"""
    uuid: str
    name: Optional[str] = None
    job_uuid: Optional[str] = None
    checkpoint_id: Optional[int] = None
    checkpoint_step: Optional[int] = None
    dataset_uuid: str
    dataset_name: Optional[str] = None
    capability: Optional[str] = None
    model_type: str = "checkpoint"
    status: str
    score: Optional[float]
    correct_count: Optional[int]
    total_count: Optional[int]
    label_results: Dict[str, Any] = {}
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]


class EvalResultsResponse(BaseModel):
    """Response for evaluation results grouped by capability"""
    job_uuid: str
    results_by_capability: Dict[str, List[Dict[str, Any]]]


# ============== Dataset Management Endpoints ==============

@router.post("/datasets/upload", response_model=EvalDatasetResponse)
async def upload_eval_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Query(..., min_length=1, max_length=100),
    format: EvalDatasetFormat = Query(...),
    capability: EvalCapability = Query(...),
    eval_method: EvalMethod = Query(EvalMethod.EXACT_MATCH),
    description: Optional[str] = Query(None),
    label_field: Optional[str] = Query(None, description="Field name for grouping results (e.g., 'difficulty', 'category')"),
    judge_model: Optional[str] = Query(None, description="LLM model for judging"),
    judge_prompt_template: Optional[str] = Query(None, description="Custom judge prompt template with {question}, {expected}, {actual} placeholders"),
    session: Session = Depends(get_session),
) -> EvalDatasetResponse:
    """
    Upload an evaluation dataset (JSONL format).

    Supported formats:
    - QA: {"question": "...", "answer": "..."}
    - Dialogue: {"messages": [{"role": "user/assistant", "content": "..."}]}
    """
    if not file.filename.endswith(('.jsonl', '.json', '.ndjson')):
        raise HTTPException(
            status_code=400,
            detail="Only .jsonl, .json, or .ndjson files are supported"
        )

    dataset_uuid = str(uuid.uuid4())[:8]

    # Ensure directory exists
    os.makedirs(EVAL_DATASETS_DIR, exist_ok=True)

    file_path = os.path.join(EVAL_DATASETS_DIR, f"{dataset_uuid}.jsonl")

    try:
        content = await file.read()

        # Validate and count samples
        lines = content.decode('utf-8').strip().split('\n')
        sample_count = 0

        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Validate format
                if format == EvalDatasetFormat.QA:
                    if 'question' not in data or 'answer' not in data:
                        raise HTTPException(
                            status_code=400,
                            detail="QA format requires 'question' and 'answer' fields"
                        )
                elif format == EvalDatasetFormat.DIALOGUE:
                    if 'messages' not in data or not isinstance(data['messages'], list):
                        raise HTTPException(
                            status_code=400,
                            detail="Dialogue format requires 'messages' array"
                        )
                sample_count += 1
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        if sample_count == 0:
            raise HTTPException(status_code=400, detail="No valid samples found")

        # Extract available labels if label_field is specified
        available_labels = []
        if label_field:
            label_values = set()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if label_field in data and data[label_field] is not None:
                        label_values.add(str(data[label_field]))
                except json.JSONDecodeError:
                    continue
            available_labels = sorted(list(label_values))

        # Save file
        with open(file_path, 'wb') as f:
            f.write(content)

        # Build judge config
        judge_config = {}
        if eval_method == EvalMethod.LLM_JUDGE:
            judge_config = {
                "model": judge_model or "qwen-plus",
                "api_key_env": "DASHSCOPE_API_KEY",
            }

        # Create database record
        repo = EvalDatasetRepository(session)
        dataset = EvalDataset(
            uuid=dataset_uuid,
            name=name,
            description=description,
            format=format,
            capability=capability,
            eval_method=eval_method,
            file_path=file_path,
            sample_count=sample_count,
            label_field=label_field,
            available_labels=available_labels,
            judge_prompt_template=judge_prompt_template,
            judge_config=judge_config,
        )
        dataset = repo.create(dataset)

        return EvalDatasetResponse(
            uuid=dataset.uuid,
            name=dataset.name,
            description=dataset.description,
            format=dataset.format.value,
            capability=dataset.capability.value,
            eval_method=dataset.eval_method.value,
            sample_count=dataset.sample_count,
            label_field=dataset.label_field,
            available_labels=dataset.available_labels,
            judge_prompt_template=dataset.judge_prompt_template,
            judge_config=dataset.judge_config,
            created_at=dataset.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", response_model=EvalDatasetListResponse)
async def list_eval_datasets(
    capability: Optional[EvalCapability] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> EvalDatasetListResponse:
    """List all evaluation datasets with optional filtering."""
    repo = EvalDatasetRepository(session)
    datasets, total = repo.list_datasets(
        capability=capability,
        offset=(page - 1) * page_size,
        limit=page_size,
    )

    return EvalDatasetListResponse(
        datasets=[
            EvalDatasetResponse(
                uuid=d.uuid,
                name=d.name,
                description=d.description,
                format=d.format.value,
                capability=d.capability.value,
                eval_method=d.eval_method.value,
                sample_count=d.sample_count,
                label_field=d.label_field,
                available_labels=d.available_labels or [],
                judge_prompt_template=d.judge_prompt_template,
                judge_config=d.judge_config,
                created_at=d.created_at,
            )
            for d in datasets
        ],
        total=total,
    )


@router.get("/datasets/{dataset_uuid}", response_model=EvalDatasetResponse)
async def get_eval_dataset(
    dataset_uuid: str,
    session: Session = Depends(get_session),
) -> EvalDatasetResponse:
    """Get evaluation dataset details."""
    repo = EvalDatasetRepository(session)
    dataset = repo.get_by_uuid(dataset_uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return EvalDatasetResponse(
        uuid=dataset.uuid,
        name=dataset.name,
        description=dataset.description,
        format=dataset.format.value,
        capability=dataset.capability.value,
        eval_method=dataset.eval_method.value,
        sample_count=dataset.sample_count,
        label_field=dataset.label_field,
        available_labels=dataset.available_labels or [],
        judge_prompt_template=dataset.judge_prompt_template,
        judge_config=dataset.judge_config,
        created_at=dataset.created_at,
    )


@router.get("/datasets/{dataset_uuid}/preview", response_model=EvalDatasetPreview)
async def preview_eval_dataset(
    dataset_uuid: str,
    limit: int = Query(10, ge=1, le=100),
    session: Session = Depends(get_session),
) -> EvalDatasetPreview:
    """Preview samples from an evaluation dataset."""
    repo = EvalDatasetRepository(session)
    dataset = repo.get_by_uuid(dataset_uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    samples = []
    try:
        with open(dataset.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                if line.strip():
                    samples.append(json.loads(line))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    return EvalDatasetPreview(
        uuid=dataset.uuid,
        name=dataset.name,
        format=dataset.format.value,
        sample_count=dataset.sample_count,
        samples=samples,
    )


@router.delete("/datasets/{dataset_uuid}")
async def delete_eval_dataset(
    dataset_uuid: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Delete an evaluation dataset."""
    repo = EvalDatasetRepository(session)
    dataset = repo.get_by_uuid(dataset_uuid)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete file
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)

    repo.delete(dataset_uuid)

    return {"success": True, "message": "Dataset deleted"}


# ============== Evaluation Trigger Endpoints ==============

@router.post("/trigger", response_model=List[EvalTaskResponse])
async def trigger_evaluation(
    request: TriggerEvalRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> List[EvalTaskResponse]:
    """
    Trigger evaluation with selected datasets.

    Supports three model types:
    - api: Use OpenAI-compatible API for inference
    - local_model: Load and run local model
    - checkpoint: Load specific training checkpoint
    """
    checkpoint_repo = CheckpointRepository(session)
    dataset_repo = EvalDatasetRepository(session)
    task_repo = EvalTaskRepository(session)

    # Build model configuration based on model_type
    model_config = {}
    checkpoint = None
    job_uuid = None
    checkpoint_step = None

    if request.model_type == ModelType.CHECKPOINT:
        if not request.checkpoint_id:
            raise HTTPException(status_code=400, detail="checkpoint_id required for checkpoint mode")
        checkpoint = checkpoint_repo.get_by_id(request.checkpoint_id)
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        model_config = {"checkpoint_path": checkpoint.path}
        job_uuid = checkpoint.job_uuid
        checkpoint_step = checkpoint.step

    elif request.model_type == ModelType.LOCAL_MODEL:
        if not request.model_path:
            raise HTTPException(status_code=400, detail="model_path required for local_model mode")
        model_config = {"model_path": request.model_path}

    elif request.model_type == ModelType.API:
        model_config = {
            "base_url": request.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": request.api_model or "qwen-plus",
            "api_key_env": request.api_key_env or "DASHSCOPE_API_KEY",
        }

    tasks = []
    for dataset_uuid in request.dataset_uuids:
        dataset = dataset_repo.get_by_uuid(dataset_uuid)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_uuid} not found")

        # Create task
        task_uuid_str = str(uuid.uuid4())[:8]
        task = EvalTask(
            uuid=task_uuid_str,
            name=request.name,
            job_uuid=job_uuid,
            checkpoint_id=checkpoint.id if checkpoint else None,
            dataset_uuid=dataset_uuid,
            model_type=request.model_type.value,
            inference_config=model_config,
            status=EvalTaskStatus.PENDING,
        )
        task = task_repo.create(task)

        # Queue background task
        background_tasks.add_task(
            _run_evaluation,
            task_uuid=task_uuid_str,
            model_type=request.model_type.value,
            model_config=model_config,
            dataset_path=dataset.file_path,
            dataset_format=dataset.format.value,
            eval_method=dataset.eval_method.value,
            judge_config=dataset.judge_config,
            label_field=dataset.label_field,
            judge_prompt_template=dataset.judge_prompt_template,
        )

        tasks.append(EvalTaskResponse(
            uuid=task.uuid,
            name=task.name,
            job_uuid=task.job_uuid,
            checkpoint_id=task.checkpoint_id,
            checkpoint_step=checkpoint_step,
            dataset_uuid=task.dataset_uuid,
            dataset_name=dataset.name,
            capability=dataset.capability.value,
            model_type=task.model_type,
            status=task.status.value,
            score=task.score,
            correct_count=task.correct_count,
            total_count=task.total_count,
            label_results=task.label_results or {},
            error_message=task.error_message,
            created_at=task.created_at,
            completed_at=task.completed_at,
        ))

    return tasks


def _call_api_model(
    messages: List[Dict[str, str]],
    model_config: Dict[str, Any],
) -> str:
    """Call OpenAI-compatible API for inference."""
    try:
        from openai import OpenAI

        api_key = os.getenv(model_config.get("api_key_env", "DASHSCOPE_API_KEY"))
        if not api_key:
            raise ValueError(f"API key not found in env: {model_config.get('api_key_env')}")

        client = OpenAI(
            api_key=api_key,
            base_url=model_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )

        response = client.chat.completions.create(
            model=model_config.get("model", "qwen-plus"),
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise


# Global vLLM engine cache
_vllm_engines: Dict[str, VLLMInferenceEngine] = {}


def _get_vllm_engine(model_path: str) -> VLLMInferenceEngine:
    """Get or create a vLLM inference engine for the given model path."""
    if model_path not in _vllm_engines:
        config = InferenceConfig(
            model_path=model_path,
            max_tokens=512,
            temperature=0.1,
        )
        engine = VLLMInferenceEngine(config)
        if engine.initialize():
            _vllm_engines[model_path] = engine
        else:
            raise RuntimeError(f"Failed to initialize vLLM engine for {model_path}")
    return _vllm_engines[model_path]


def _call_vllm_model(
    messages: List[Dict[str, str]],
    model_config: Dict[str, Any],
) -> str:
    """
    Call local model using vLLM for inference.

    Args:
        messages: Chat messages
        model_config: Must contain 'model_path' or 'checkpoint_path'

    Returns:
        Generated response
    """
    model_path = model_config.get("model_path") or model_config.get("checkpoint_path")
    if not model_path:
        raise ValueError("model_path or checkpoint_path required for vLLM inference")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    try:
        engine = _get_vllm_engine(model_path)
        return engine.generate(messages)
    except Exception as e:
        logger.error(f"vLLM inference failed: {e}")
        raise


def _evaluate_answer(
    model_output: str,
    expected: str,
    eval_method: str,
    judge_config: Dict[str, Any] = None,
    judge_prompt_template: str = None,
    question: str = None,
) -> bool:
    """Evaluate if model output matches expected answer."""
    if eval_method == "exact_match":
        return model_output.strip().lower() == expected.strip().lower()

    elif eval_method == "contains":
        # Support multiple keywords separated by |
        keywords = expected.split("|")
        return any(kw.strip().lower() in model_output.lower() for kw in keywords)

    elif eval_method == "llm_judge":
        # Use LLM to judge answer correctness
        try:
            # Use custom prompt template if provided
            if judge_prompt_template:
                prompt = judge_prompt_template.format(
                    question=question or "",
                    expected=expected,
                    actual=model_output,
                )
            else:
                prompt = f"""判断模型回答是否正确。

预期答案: {expected}
模型回答: {model_output}

只回复 YES 或 NO。"""

            judge_messages = [{"role": "user", "content": prompt}]
            result = _call_api_model(judge_messages, judge_config or {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-plus",
                "api_key_env": "DASHSCOPE_API_KEY",
            })
            return result.upper().startswith("YES")
        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            return False

    return False


async def _run_evaluation(
    task_uuid: str,
    model_type: str,
    model_config: Dict[str, Any],
    dataset_path: str,
    dataset_format: str,
    eval_method: str,
    judge_config: Dict[str, Any],
    label_field: Optional[str] = None,
    judge_prompt_template: Optional[str] = None,
):
    """
    Background task to run evaluation.

    Supports:
    - API-based inference (OpenAI compatible)
    - Local model inference (placeholder - needs vLLM/transformers)
    - Checkpoint inference (placeholder - needs model loading)
    - Per-label result tracking
    """
    with Session(engine) as session:
        repo = EvalTaskRepository(session)
        task = repo.get_by_uuid(task_uuid)

        if not task:
            logger.error(f"Task {task_uuid} not found")
            return

        try:
            task.status = EvalTaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            repo.update(task)

            # Load dataset
            samples = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

            results = []
            sample_results = []
            correct = 0

            # Track per-label results
            label_stats = {}  # {label: {"correct": 0, "total": 0}}

            for i, sample in enumerate(samples):
                try:
                    # Get label for this sample
                    sample_label = None
                    if label_field and label_field in sample:
                        sample_label = str(sample[label_field])
                        if sample_label not in label_stats:
                            label_stats[sample_label] = {"correct": 0, "total": 0}
                        label_stats[sample_label]["total"] += 1

                    # Build messages for inference
                    question = ""
                    if dataset_format == "qa":
                        question = sample.get("question", "")
                        messages = [{"role": "user", "content": question}]
                        expected = sample.get("answer", "")
                    else:  # dialogue
                        messages = sample.get("messages", [])
                        # For dialogue, expected is the last assistant message
                        expected = ""
                        for msg in reversed(messages):
                            if msg.get("role") == "assistant":
                                expected = msg.get("content", "")
                                break
                        # Get last user message as question
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                question = msg.get("content", "")
                                break
                        # Remove last assistant message for inference
                        messages = [m for m in messages if not (m == messages[-1] and m.get("role") == "assistant")]

                    # Run inference based on model_type
                    if model_type == "api":
                        model_output = _call_api_model(messages, model_config)
                    elif model_type == "local_model":
                        try:
                            model_output = _call_vllm_model(messages, model_config)
                        except Exception as e:
                            logger.error(f"Local model inference failed: {e}")
                            model_output = f"[Local model inference error: {str(e)}]"
                    elif model_type == "checkpoint":
                        try:
                            model_output = _call_vllm_model(messages, model_config)
                        except Exception as e:
                            logger.error(f"Checkpoint inference failed: {e}")
                            model_output = f"[Checkpoint inference error: {str(e)}]"
                    else:
                        model_output = "[Unknown model type]"

                    # Evaluate answer
                    is_correct = _evaluate_answer(
                        model_output, expected, eval_method, judge_config,
                        judge_prompt_template=judge_prompt_template,
                        question=question,
                    )
                    if is_correct:
                        correct += 1
                        if sample_label and sample_label in label_stats:
                            label_stats[sample_label]["correct"] += 1

                    # Store sample result for comparison
                    sample_result = {
                        "id": sample.get("id", str(i)),
                        "input": question[:500] if question else str(messages)[:500],
                        "expected": expected[:500],
                        "actual": model_output[:500],
                        "correct": is_correct,
                    }
                    if sample_label:
                        sample_result["label"] = sample_label

                    sample_results.append(sample_result)

                    results.append({
                        "input": question[:200] if question else str(messages)[:200],
                        "expected": expected[:200],
                        "actual": model_output[:500],
                        "correct": is_correct,
                        "label": sample_label,
                    })

                    # Log progress every 10 samples
                    if (i + 1) % 10 == 0:
                        logger.info(f"Task {task_uuid}: {i + 1}/{len(samples)} samples processed")

                except Exception as e:
                    logger.error(f"Sample evaluation failed: {e}")
                    results.append({
                        "input": str(sample)[:200],
                        "expected": "",
                        "actual": f"Error: {str(e)}",
                        "correct": False,
                    })

            # Compute per-label accuracy
            label_results = {}
            for label, stats in label_stats.items():
                label_results[label] = {
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "accuracy": round((stats["correct"] / stats["total"]) * 100, 2) if stats["total"] > 0 else 0,
                }

            task.score = (correct / len(samples)) * 100 if samples else 0
            task.correct_count = correct
            task.total_count = len(samples)
            task.detailed_results = {"samples": results[:100]}  # Store first 100
            task.label_results = label_results
            task.sample_results = sample_results  # Store all for comparison
            task.status = EvalTaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()

        except Exception as e:
            logger.error(f"Evaluation failed for task {task_uuid}: {e}")
            task.status = EvalTaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()

        repo.update(task)


# ============== Results Endpoints ==============

@router.get("/tasks/{task_uuid}", response_model=EvalTaskResponse)
async def get_eval_task(
    task_uuid: str,
    session: Session = Depends(get_session),
) -> EvalTaskResponse:
    """Get evaluation task status and results."""
    task_repo = EvalTaskRepository(session)
    checkpoint_repo = CheckpointRepository(session)
    dataset_repo = EvalDatasetRepository(session)

    task = task_repo.get_by_uuid(task_uuid)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Get checkpoint step
    checkpoint = checkpoint_repo.get_by_id(task.checkpoint_id)

    # Get dataset info
    dataset = dataset_repo.get_by_uuid(task.dataset_uuid)

    return EvalTaskResponse(
        uuid=task.uuid,
        name=task.name,
        job_uuid=task.job_uuid,
        checkpoint_id=task.checkpoint_id,
        checkpoint_step=checkpoint.step if checkpoint else None,
        dataset_uuid=task.dataset_uuid,
        dataset_name=dataset.name if dataset else None,
        capability=dataset.capability.value if dataset else None,
        model_type=task.model_type,
        status=task.status.value,
        score=task.score,
        correct_count=task.correct_count,
        total_count=task.total_count,
        label_results=task.label_results or {},
        error_message=task.error_message,
        created_at=task.created_at,
        completed_at=task.completed_at,
    )


@router.get("/tasks/{task_uuid}/details")
async def get_eval_task_details(
    task_uuid: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Get detailed evaluation results including per-sample results."""
    task_repo = EvalTaskRepository(session)
    task = task_repo.get_by_uuid(task_uuid)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "uuid": task.uuid,
        "name": task.name,
        "status": task.status.value,
        "score": task.score,
        "label_results": task.label_results or {},
        "detailed_results": task.detailed_results,
        "sample_results": task.sample_results or [],
    }


@router.get("/jobs/{job_uuid}/results", response_model=EvalResultsResponse)
async def get_job_eval_results(
    job_uuid: str,
    session: Session = Depends(get_session),
) -> EvalResultsResponse:
    """
    Get all evaluation results for a job, grouped by capability.

    Useful for plotting scores over training steps.
    """
    task_repo = EvalTaskRepository(session)
    checkpoint_repo = CheckpointRepository(session)
    dataset_repo = EvalDatasetRepository(session)

    tasks = task_repo.get_tasks_for_job(job_uuid)
    checkpoints = checkpoint_repo.get_checkpoints(job_uuid)
    checkpoint_map = {c.id: c.step for c in checkpoints}

    results_by_capability: Dict[str, List[Dict[str, Any]]] = {}

    for task in tasks:
        if task.status != EvalTaskStatus.COMPLETED:
            continue

        dataset = dataset_repo.get_by_uuid(task.dataset_uuid)
        if not dataset:
            continue

        capability = dataset.capability.value
        if capability not in results_by_capability:
            results_by_capability[capability] = []

        results_by_capability[capability].append({
            "step": checkpoint_map.get(task.checkpoint_id, 0),
            "score": task.score,
            "dataset_name": dataset.name,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        })

    # Sort each capability's results by step
    for capability in results_by_capability:
        results_by_capability[capability].sort(key=lambda x: x["step"])

    return EvalResultsResponse(
        job_uuid=job_uuid,
        results_by_capability=results_by_capability,
    )


@router.get("/jobs/{job_uuid}/tasks", response_model=List[EvalTaskResponse])
async def get_job_eval_tasks(
    job_uuid: str,
    status: Optional[EvalTaskStatus] = Query(None),
    session: Session = Depends(get_session),
) -> List[EvalTaskResponse]:
    """Get all evaluation tasks for a job."""
    task_repo = EvalTaskRepository(session)
    checkpoint_repo = CheckpointRepository(session)
    dataset_repo = EvalDatasetRepository(session)

    tasks = task_repo.get_tasks_for_job(job_uuid)
    checkpoints = checkpoint_repo.get_checkpoints(job_uuid)
    checkpoint_map = {c.id: c.step for c in checkpoints}

    if status:
        tasks = [t for t in tasks if t.status == status]

    responses = []
    for task in tasks:
        dataset = dataset_repo.get_by_uuid(task.dataset_uuid)
        responses.append(EvalTaskResponse(
            uuid=task.uuid,
            name=task.name,
            job_uuid=task.job_uuid,
            checkpoint_id=task.checkpoint_id,
            checkpoint_step=checkpoint_map.get(task.checkpoint_id),
            dataset_uuid=task.dataset_uuid,
            dataset_name=dataset.name if dataset else None,
            capability=dataset.capability.value if dataset else None,
            model_type=task.model_type,
            status=task.status.value,
            score=task.score,
            correct_count=task.correct_count,
            total_count=task.total_count,
            label_results=task.label_results or {},
            error_message=task.error_message,
            created_at=task.created_at,
            completed_at=task.completed_at,
        ))

    return responses


# ============== Comparison Endpoints ==============

class CreateComparisonRequest(BaseModel):
    """Request to create a comparison"""
    name: str
    description: Optional[str] = None
    dataset_uuid: str
    model_a_task_uuid: str
    model_a_name: str  # e.g., "训练前", "Baseline"
    model_b_task_uuid: str
    model_b_name: str  # e.g., "训练后", "Step 1000"


class ComparisonResponse(BaseModel):
    """Response for comparison"""
    uuid: str
    name: str
    description: Optional[str] = None
    dataset_uuid: str
    dataset_name: Optional[str] = None
    model_a_task_uuid: str
    model_a_name: str
    model_a_score: Optional[float] = None
    model_b_task_uuid: str
    model_b_name: str
    model_b_score: Optional[float] = None
    comparison_results: Dict[str, Any]
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None


class SampleDiff(BaseModel):
    """Sample-level comparison diff"""
    id: str
    input: str
    expected: str
    model_a_output: str
    model_a_correct: bool
    model_b_output: str
    model_b_correct: bool
    change: str  # "improved", "degraded", "unchanged"
    label: Optional[str] = None


@router.post("/comparisons", response_model=ComparisonResponse)
async def create_comparison(
    request: CreateComparisonRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> ComparisonResponse:
    """
    Create a comparison between two evaluation tasks.

    Compares the results of two evaluation runs on the same dataset
    to identify improvements and regressions.
    """
    task_repo = EvalTaskRepository(session)
    dataset_repo = EvalDatasetRepository(session)
    comparison_repo = EvalComparisonRepository(session)

    # Validate tasks exist and are completed
    task_a = task_repo.get_by_uuid(request.model_a_task_uuid)
    task_b = task_repo.get_by_uuid(request.model_b_task_uuid)

    if not task_a:
        raise HTTPException(status_code=404, detail=f"Task A not found: {request.model_a_task_uuid}")
    if not task_b:
        raise HTTPException(status_code=404, detail=f"Task B not found: {request.model_b_task_uuid}")

    if task_a.status != EvalTaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task A is not completed")
    if task_b.status != EvalTaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task B is not completed")

    # Validate dataset
    dataset = dataset_repo.get_by_uuid(request.dataset_uuid)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create comparison
    comparison_uuid = str(uuid.uuid4())[:8]
    comparison = EvalComparison(
        uuid=comparison_uuid,
        name=request.name,
        description=request.description,
        dataset_uuid=request.dataset_uuid,
        model_a_task_uuid=request.model_a_task_uuid,
        model_a_name=request.model_a_name,
        model_b_task_uuid=request.model_b_task_uuid,
        model_b_name=request.model_b_name,
        status=EvalTaskStatus.PENDING,
    )
    comparison = comparison_repo.create(comparison)

    # Run comparison in background
    background_tasks.add_task(
        _run_comparison,
        comparison_uuid=comparison_uuid,
    )

    return ComparisonResponse(
        uuid=comparison.uuid,
        name=comparison.name,
        description=comparison.description,
        dataset_uuid=comparison.dataset_uuid,
        dataset_name=dataset.name,
        model_a_task_uuid=comparison.model_a_task_uuid,
        model_a_name=comparison.model_a_name,
        model_a_score=task_a.score,
        model_b_task_uuid=comparison.model_b_task_uuid,
        model_b_name=comparison.model_b_name,
        model_b_score=task_b.score,
        comparison_results=comparison.comparison_results,
        status=comparison.status.value,
        created_at=comparison.created_at,
        completed_at=comparison.completed_at,
    )


async def _run_comparison(comparison_uuid: str):
    """Background task to compute comparison results."""
    with Session(engine) as session:
        comparison_repo = EvalComparisonRepository(session)
        task_repo = EvalTaskRepository(session)

        comparison = comparison_repo.get_by_uuid(comparison_uuid)
        if not comparison:
            logger.error(f"Comparison {comparison_uuid} not found")
            return

        try:
            comparison.status = EvalTaskStatus.RUNNING
            comparison_repo.update(comparison)

            # Get tasks
            task_a = task_repo.get_by_uuid(comparison.model_a_task_uuid)
            task_b = task_repo.get_by_uuid(comparison.model_b_task_uuid)

            if not task_a or not task_b:
                raise ValueError("One or both tasks not found")

            # Get sample results
            samples_a = {s.get("id", str(i)): s for i, s in enumerate(task_a.sample_results or [])}
            samples_b = {s.get("id", str(i)): s for i, s in enumerate(task_b.sample_results or [])}

            # Compute diffs
            sample_diffs = []
            improved_count = 0
            degraded_count = 0
            unchanged_count = 0

            # Track per-label changes
            label_changes = {}  # {label: {"improved": 0, "degraded": 0, "unchanged": 0}}

            all_ids = set(samples_a.keys()) | set(samples_b.keys())

            for sample_id in all_ids:
                sample_a = samples_a.get(sample_id, {})
                sample_b = samples_b.get(sample_id, {})

                correct_a = sample_a.get("correct", False)
                correct_b = sample_b.get("correct", False)

                # Determine change
                if not correct_a and correct_b:
                    change = "improved"
                    improved_count += 1
                elif correct_a and not correct_b:
                    change = "degraded"
                    degraded_count += 1
                else:
                    change = "unchanged"
                    unchanged_count += 1

                # Track per-label
                label = sample_a.get("label") or sample_b.get("label")
                if label:
                    if label not in label_changes:
                        label_changes[label] = {"improved": 0, "degraded": 0, "unchanged": 0}
                    label_changes[label][change] += 1

                sample_diffs.append({
                    "id": sample_id,
                    "input": sample_a.get("input", sample_b.get("input", ""))[:500],
                    "expected": sample_a.get("expected", sample_b.get("expected", ""))[:500],
                    "model_a_output": sample_a.get("actual", "")[:500],
                    "model_a_correct": correct_a,
                    "model_b_output": sample_b.get("actual", "")[:500],
                    "model_b_correct": correct_b,
                    "change": change,
                    "label": label,
                })

            # Compute overall stats
            comparison_results = {
                "overall": {
                    "model_a_accuracy": task_a.score,
                    "model_b_accuracy": task_b.score,
                    "delta": round((task_b.score or 0) - (task_a.score or 0), 2),
                },
                "by_label": {},
                "improved_count": improved_count,
                "degraded_count": degraded_count,
                "unchanged_count": unchanged_count,
            }

            # Add per-label deltas
            label_results_a = task_a.label_results or {}
            label_results_b = task_b.label_results or {}

            all_labels = set(label_results_a.keys()) | set(label_results_b.keys())
            for label in all_labels:
                acc_a = label_results_a.get(label, {}).get("accuracy", 0)
                acc_b = label_results_b.get(label, {}).get("accuracy", 0)
                comparison_results["by_label"][label] = {
                    "model_a_accuracy": acc_a,
                    "model_b_accuracy": acc_b,
                    "delta": round(acc_b - acc_a, 2),
                    "changes": label_changes.get(label, {}),
                }

            comparison.comparison_results = comparison_results
            comparison.sample_diffs = sample_diffs
            comparison.status = EvalTaskStatus.COMPLETED
            comparison.completed_at = datetime.utcnow()

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            comparison.status = EvalTaskStatus.FAILED
            comparison.completed_at = datetime.utcnow()

        comparison_repo.update(comparison)


@router.get("/comparisons", response_model=List[ComparisonResponse])
async def list_comparisons(
    dataset_uuid: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> List[ComparisonResponse]:
    """List all comparisons with optional filtering by dataset."""
    comparison_repo = EvalComparisonRepository(session)
    dataset_repo = EvalDatasetRepository(session)
    task_repo = EvalTaskRepository(session)

    comparisons, total = comparison_repo.list_comparisons(
        dataset_uuid=dataset_uuid,
        offset=offset,
        limit=limit,
    )

    responses = []
    for c in comparisons:
        dataset = dataset_repo.get_by_uuid(c.dataset_uuid)
        task_a = task_repo.get_by_uuid(c.model_a_task_uuid)
        task_b = task_repo.get_by_uuid(c.model_b_task_uuid)

        responses.append(ComparisonResponse(
            uuid=c.uuid,
            name=c.name,
            description=c.description,
            dataset_uuid=c.dataset_uuid,
            dataset_name=dataset.name if dataset else None,
            model_a_task_uuid=c.model_a_task_uuid,
            model_a_name=c.model_a_name,
            model_a_score=task_a.score if task_a else None,
            model_b_task_uuid=c.model_b_task_uuid,
            model_b_name=c.model_b_name,
            model_b_score=task_b.score if task_b else None,
            comparison_results=c.comparison_results,
            status=c.status.value,
            created_at=c.created_at,
            completed_at=c.completed_at,
        ))

    return responses


@router.get("/comparisons/{comparison_uuid}", response_model=ComparisonResponse)
async def get_comparison(
    comparison_uuid: str,
    session: Session = Depends(get_session),
) -> ComparisonResponse:
    """Get comparison details."""
    comparison_repo = EvalComparisonRepository(session)
    dataset_repo = EvalDatasetRepository(session)
    task_repo = EvalTaskRepository(session)

    comparison = comparison_repo.get_by_uuid(comparison_uuid)
    if not comparison:
        raise HTTPException(status_code=404, detail="Comparison not found")

    dataset = dataset_repo.get_by_uuid(comparison.dataset_uuid)
    task_a = task_repo.get_by_uuid(comparison.model_a_task_uuid)
    task_b = task_repo.get_by_uuid(comparison.model_b_task_uuid)

    return ComparisonResponse(
        uuid=comparison.uuid,
        name=comparison.name,
        description=comparison.description,
        dataset_uuid=comparison.dataset_uuid,
        dataset_name=dataset.name if dataset else None,
        model_a_task_uuid=comparison.model_a_task_uuid,
        model_a_name=comparison.model_a_name,
        model_a_score=task_a.score if task_a else None,
        model_b_task_uuid=comparison.model_b_task_uuid,
        model_b_name=comparison.model_b_name,
        model_b_score=task_b.score if task_b else None,
        comparison_results=comparison.comparison_results,
        status=comparison.status.value,
        created_at=comparison.created_at,
        completed_at=comparison.completed_at,
    )


@router.get("/comparisons/{comparison_uuid}/diffs", response_model=List[SampleDiff])
async def get_comparison_diffs(
    comparison_uuid: str,
    change_filter: Optional[str] = Query(None, description="Filter by change type: improved, degraded, unchanged"),
    label_filter: Optional[str] = Query(None, description="Filter by label"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> List[SampleDiff]:
    """Get sample-level diffs for a comparison."""
    comparison_repo = EvalComparisonRepository(session)

    comparison = comparison_repo.get_by_uuid(comparison_uuid)
    if not comparison:
        raise HTTPException(status_code=404, detail="Comparison not found")

    diffs = comparison.sample_diffs or []

    # Apply filters
    if change_filter:
        diffs = [d for d in diffs if d.get("change") == change_filter]
    if label_filter:
        diffs = [d for d in diffs if d.get("label") == label_filter]

    # Apply pagination
    paginated = diffs[offset:offset + limit]

    return [SampleDiff(**d) for d in paginated]


@router.delete("/comparisons/{comparison_uuid}")
async def delete_comparison(
    comparison_uuid: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Delete a comparison."""
    comparison_repo = EvalComparisonRepository(session)

    comparison = comparison_repo.get_by_uuid(comparison_uuid)
    if not comparison:
        raise HTTPException(status_code=404, detail="Comparison not found")

    comparison_repo.delete(comparison_uuid)

    return {"success": True, "message": "Comparison deleted"}


@router.get("/tasks", response_model=List[EvalTaskResponse])
async def list_eval_tasks(
    dataset_uuid: Optional[str] = Query(None),
    status: Optional[EvalTaskStatus] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    session: Session = Depends(get_session),
) -> List[EvalTaskResponse]:
    """
    List all evaluation tasks with optional filtering.

    Useful for selecting tasks when creating comparisons.
    """
    task_repo = EvalTaskRepository(session)
    dataset_repo = EvalDatasetRepository(session)
    checkpoint_repo = CheckpointRepository(session)

    if dataset_uuid:
        tasks = task_repo.get_tasks_for_dataset(dataset_uuid)
        if status:
            tasks = [t for t in tasks if t.status == status]
        tasks = tasks[offset:offset + limit]
        total = len(tasks)
    else:
        tasks, total = task_repo.list_tasks(status=status, offset=offset, limit=limit)

    responses = []
    for task in tasks:
        dataset = dataset_repo.get_by_uuid(task.dataset_uuid)
        checkpoint = checkpoint_repo.get_by_id(task.checkpoint_id) if task.checkpoint_id else None

        responses.append(EvalTaskResponse(
            uuid=task.uuid,
            name=task.name,
            job_uuid=task.job_uuid,
            checkpoint_id=task.checkpoint_id,
            checkpoint_step=checkpoint.step if checkpoint else None,
            dataset_uuid=task.dataset_uuid,
            dataset_name=dataset.name if dataset else None,
            capability=dataset.capability.value if dataset else None,
            model_type=task.model_type,
            status=task.status.value,
            score=task.score,
            correct_count=task.correct_count,
            total_count=task.total_count,
            label_results=task.label_results or {},
            error_message=task.error_message,
            created_at=task.created_at,
            completed_at=task.completed_at,
        ))

    return responses
