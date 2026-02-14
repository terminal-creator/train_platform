"""
Data Factory API Router

Provides endpoints for:
- Data cleaning pipeline
- Deduplication (MinHash/SimHash)
- Quality assessment
- Format conversion
- Data splitting
- Config templates
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import json
import os

from sqlmodel import Session
from ...core.database import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-factory", tags=["Data Factory"])

DATASETS_DIR = os.environ.get("DATASETS_DIR", "./datasets")


# ============== Request/Response Models ==============

class CleaningRequest(BaseModel):
    """Request for data cleaning."""
    input_path: str
    output_path: Optional[str] = None
    min_prompt_length: int = 1
    max_prompt_length: int = 10000
    min_response_length: int = 1
    max_response_length: int = 50000
    min_prompt_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    remove_empty: bool = True
    remove_duplicates: bool = False
    strip_whitespace: bool = True
    remove_html_tags: bool = False
    min_unique_chars_ratio: float = 0.0
    max_repetition_ratio: float = 1.0
    required_fields: List[str] = []


class DeduplicationRequest(BaseModel):
    """Request for deduplication."""
    input_path: str
    output_path: Optional[str] = None
    method: str = "minhash"  # minhash, simhash, exact
    threshold: float = 0.8
    num_perm: int = 128
    ngram_size: int = 3
    text_fields: List[str] = ["prompt", "response"]


class QualityAssessmentRequest(BaseModel):
    """Request for quality assessment."""
    input_path: str
    prompt_field: str = "prompt"
    response_field: str = "response"


class FormatConversionRequest(BaseModel):
    """Request for format conversion."""
    input_path: str
    output_path: str
    target_format: str  # sft, dpo, grpo, ppo, openai_messages
    source_format: Optional[str] = None  # auto-detect if not provided


class SplitRequest(BaseModel):
    """Request for data splitting."""
    input_path: str
    output_dir: str
    method: str = "random"  # random, stratified, temporal
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    category_field: Optional[str] = None
    time_field: Optional[str] = None
    output_format: str = "jsonl"


class FormatDetectRequest(BaseModel):
    """Request for format detection."""
    input_path: str


# ============== Config Templates Endpoints ==============

@router.get("/templates")
async def list_config_templates() -> List[Dict[str, Any]]:
    """List all available training config templates."""
    from ...core.config_templates import list_templates
    return list_templates()


@router.get("/templates/{algorithm}")
async def get_config_template(algorithm: str) -> Dict[str, Any]:
    """Get a config template for a specific algorithm."""
    from ...core.config_templates import get_template
    try:
        return get_template(algorithm)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template not found: {algorithm}")


@router.post("/templates/{algorithm}/validate")
async def validate_config(
    algorithm: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a user config against the template."""
    from ...core.config_templates import validate_config as _validate
    try:
        return _validate(algorithm, config)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template not found: {algorithm}")


# ============== Format Detection ==============

@router.post("/detect-format")
async def detect_format(request: FormatDetectRequest) -> Dict[str, Any]:
    """Auto-detect data format from file."""
    from ...core.data_converter import DataFormatDetector
    try:
        fmt, confidence = DataFormatDetector.detect_from_file(request.input_path)
        return {
            "format": fmt,
            "confidence": round(confidence, 3),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============== Format Conversion ==============

@router.post("/convert")
async def convert_format(request: FormatConversionRequest) -> Dict[str, Any]:
    """Convert data between formats."""
    from ...core.data_converter import convert_file
    try:
        result = convert_file(
            input_path=request.input_path,
            output_path=request.output_path,
            target_format=request.target_format,
            source_format=request.source_format,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Data Cleaning ==============

@router.post("/clean")
async def clean_data(request: CleaningRequest) -> Dict[str, Any]:
    """Run data cleaning pipeline."""
    from ...core.data_cleaning import DataCleaningPipeline, CleaningConfig
    from ...core.data_converter import _load_data, _save_data

    try:
        data = _load_data(request.input_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load data: {e}")

    config = CleaningConfig(
        min_prompt_length=request.min_prompt_length,
        max_prompt_length=request.max_prompt_length,
        min_response_length=request.min_response_length,
        max_response_length=request.max_response_length,
        min_prompt_tokens=request.min_prompt_tokens,
        max_prompt_tokens=request.max_prompt_tokens,
        remove_empty=request.remove_empty,
        remove_duplicates=request.remove_duplicates,
        strip_whitespace=request.strip_whitespace,
        remove_html_tags=request.remove_html_tags,
        min_unique_chars_ratio=request.min_unique_chars_ratio,
        max_repetition_ratio=request.max_repetition_ratio,
        required_fields=request.required_fields,
    )

    pipeline = DataCleaningPipeline(config)
    cleaned, stats = pipeline.clean(data)

    output_path = request.output_path or request.input_path.replace(".", "_cleaned.", 1)
    _save_data(cleaned, output_path)

    return {
        "output_path": output_path,
        **stats.to_dict(),
    }


# ============== Deduplication ==============

@router.post("/deduplicate")
async def deduplicate_data(request: DeduplicationRequest) -> Dict[str, Any]:
    """Run deduplication on data."""
    from ...core.deduplication import deduplicate, DeduplicationConfig
    from ...core.data_converter import _load_data, _save_data

    try:
        data = _load_data(request.input_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load data: {e}")

    config = DeduplicationConfig(
        method=request.method,
        threshold=request.threshold,
        num_perm=request.num_perm,
        ngram_size=request.ngram_size,
        text_fields=request.text_fields,
    )

    deduped, stats = deduplicate(data, config)

    output_path = request.output_path or request.input_path.replace(".", "_deduped.", 1)
    _save_data(deduped, output_path)

    return {
        "output_path": output_path,
        **stats.to_dict(),
    }


# ============== Quality Assessment ==============

@router.post("/assess-quality")
async def assess_quality(request: QualityAssessmentRequest) -> Dict[str, Any]:
    """Run quality assessment on data."""
    from ...core.quality_assessment import QualityAssessor, QualityConfig
    from ...core.data_converter import _load_data

    try:
        data = _load_data(request.input_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load data: {e}")

    config = QualityConfig(
        prompt_field=request.prompt_field,
        response_field=request.response_field,
    )

    assessor = QualityAssessor(config)
    report = assessor.assess_dataset(data)

    return report.to_dict()


# ============== Data Splitting ==============

@router.post("/split")
async def split_data(request: SplitRequest) -> Dict[str, Any]:
    """Split data into train/val/test sets."""
    from ...core.data_splitter import split_and_save
    from ...core.data_converter import _load_data

    try:
        data = _load_data(request.input_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load data: {e}")

    try:
        result = split_and_save(
            data=data,
            output_dir=request.output_dir,
            method=request.method,
            output_format=request.output_format,
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio,
            seed=request.seed,
            category_field=request.category_field,
            time_field=request.time_field,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Split failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Benchmark Endpoints ==============

@router.get("/benchmarks")
async def list_benchmarks() -> List[Dict[str, str]]:
    """List available evaluation benchmarks."""
    from ...core.evaluation.benchmarks import list_benchmarks
    return list_benchmarks()


@router.get("/benchmarks/{benchmark}/questions")
async def get_benchmark_questions(
    benchmark: str,
    split: str = Query("test"),
    limit: int = Query(10, ge=1, le=1000),
) -> Dict[str, Any]:
    """Get questions from a benchmark for evaluation."""
    from ...core.evaluation.benchmarks import get_evaluator
    try:
        evaluator = get_evaluator(benchmark)
        questions = evaluator.get_questions(split=split, limit=limit)
        return {
            "benchmark": benchmark,
            "split": split,
            "total": len(questions),
            "questions": questions,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/benchmarks/{benchmark}/evaluate")
async def evaluate_benchmark(
    benchmark: str,
    responses: List[Dict[str, str]],
    split: str = Query("test"),
) -> Dict[str, Any]:
    """
    Evaluate responses against a benchmark.

    Each response should have: {"question_id": "...", "response": "..."}
    """
    from ...core.evaluation.benchmarks import get_evaluator
    try:
        evaluator = get_evaluator(benchmark)
        result = evaluator.evaluate_all(responses, split=split)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/benchmarks/report")
async def generate_eval_report(
    results: List[Dict[str, Any]],
    model_name: str = Query("Model"),
    output_format: str = Query("markdown"),
    comparison: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate an evaluation report."""
    from ...core.evaluation.report_generator import generate_report
    try:
        content = generate_report(results, model_name, output_format, comparison)
        return {
            "format": output_format,
            "content": content,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
