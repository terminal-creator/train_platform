"""
Dataset Version API Router (Phase 2)

提供数据版本化和血缘追踪的 REST API 接口。
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from sqlmodel import Session

from ...core.dataset_version import (
    create_dataset_snapshot,
    calculate_file_hash,
    compare_dataset_versions,
    DatasetVersionManager,
)
from ...core.database import (
    get_session,
    DatasetVersion,
    DatasetVersionRepository,
    TrainingJob,
)

router = APIRouter(prefix="/dataset-versions", tags=["Dataset Versioning"])


# ============== Request/Response Models ==============

class CreateSnapshotRequest(BaseModel):
    """创建数据集快照请求"""
    file_path: str = Field(..., description="数据集文件路径")
    dataset_name: Optional[str] = Field(None, description="数据集名称（可选）")
    description: Optional[str] = Field(None, description="描述")
    tags: List[str] = Field(default_factory=list, description="标签列表")


class DatasetSnapshotResponse(BaseModel):
    """数据集快照响应"""
    dataset_name: str
    file_path: str
    file_hash: str
    hash_algorithm: str
    file_size: int
    file_size_mb: float
    format: str
    num_samples: Optional[int]
    description: Optional[str]
    tags: List[str]
    created_at: str
    modified_at: str


class DatasetVersionDetail(BaseModel):
    """数据集版本详情"""
    id: int
    dataset_name: str
    file_path: str
    file_hash: str
    file_size_mb: float
    format: str
    num_samples: Optional[int]
    description: Optional[str]
    tags: List[str]
    created_at: str
    modified_at: str


class DatasetVersionListResponse(BaseModel):
    """数据集版本列表响应"""
    versions: List[DatasetVersionDetail]
    total: int


class DatasetLineageResponse(BaseModel):
    """数据集血缘追踪响应"""
    dataset_version: DatasetVersionDetail
    used_by_jobs: List[Dict[str, Any]]
    num_jobs: int


class CompareVersionsRequest(BaseModel):
    """对比数据集版本请求"""
    hash_a: str = Field(..., description="版本 A 的 hash")
    hash_b: str = Field(..., description="版本 B 的 hash")


class CompareVersionsResponse(BaseModel):
    """对比数据集版本响应"""
    identical: bool
    hash_changed: bool
    size_changed: bool
    samples_changed: bool
    hash_a: str
    hash_b: str
    size_diff_mb: float
    samples_diff: int


# ============== API Endpoints ==============

@router.post("/snapshot", response_model=DatasetSnapshotResponse)
async def create_snapshot(
    request: CreateSnapshotRequest,
    session: Session = Depends(get_session),
) -> DatasetSnapshotResponse:
    """
    为数据集创建版本快照

    计算数据集的 hash 并保存到数据库，用于版本追踪和血缘分析。

    Args:
        request: 创建快照请求
        session: 数据库会话

    Returns:
        数据集快照信息
    """
    try:
        # 创建快照
        snapshot = create_dataset_snapshot(
            file_path=request.file_path,
            dataset_name=request.dataset_name,
            description=request.description,
            tags=request.tags,
        )

        # 保存到数据库
        repo = DatasetVersionRepository(session)

        # 检查是否已存在相同 hash
        existing = repo.get_by_hash(snapshot["file_hash"])
        if existing:
            # 返回已存在的版本
            return DatasetSnapshotResponse(**{
                "dataset_name": existing.dataset_name,
                "file_path": existing.file_path,
                "file_hash": existing.file_hash,
                "hash_algorithm": existing.hash_algorithm,
                "file_size": existing.file_size,
                "file_size_mb": existing.file_size_mb,
                "format": existing.format,
                "num_samples": existing.num_samples,
                "description": existing.description,
                "tags": existing.tags,
                "created_at": existing.created_at.isoformat(),
                "modified_at": existing.modified_at,
            })

        # 创建新版本
        version = DatasetVersion(
            dataset_name=snapshot["dataset_name"],
            file_path=snapshot["file_path"],
            file_hash=snapshot["file_hash"],
            hash_algorithm=snapshot["hash_algorithm"],
            file_size=snapshot["file_size"],
            file_size_mb=snapshot["file_size_mb"],
            format=snapshot["format"],
            num_samples=snapshot["num_samples"],
            description=snapshot["description"],
            tags=snapshot["tags"],
            modified_at=snapshot["modified_at"],
        )

        repo.create(version)

        return DatasetSnapshotResponse(**snapshot)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建快照失败: {str(e)}")


@router.get("", response_model=DatasetVersionListResponse)
async def list_versions(
    dataset_name: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
    session: Session = Depends(get_session),
) -> DatasetVersionListResponse:
    """
    列出数据集版本

    可以按数据集名称筛选，查看某个数据集的所有版本历史。

    Args:
        dataset_name: 数据集名称（可选）
        offset: 分页偏移
        limit: 分页大小
        session: 数据库会话

    Returns:
        数据集版本列表
    """
    repo = DatasetVersionRepository(session)
    versions, total = repo.list_versions(
        dataset_name=dataset_name,
        offset=offset,
        limit=limit,
    )

    return DatasetVersionListResponse(
        versions=[
            DatasetVersionDetail(
                id=v.id,
                dataset_name=v.dataset_name,
                file_path=v.file_path,
                file_hash=v.file_hash,
                file_size_mb=v.file_size_mb,
                format=v.format,
                num_samples=v.num_samples,
                description=v.description,
                tags=v.tags,
                created_at=v.created_at.isoformat(),
                modified_at=v.modified_at,
            )
            for v in versions
        ],
        total=total,
    )


@router.get("/{file_hash}", response_model=DatasetVersionDetail)
async def get_version_by_hash(
    file_hash: str,
    session: Session = Depends(get_session),
) -> DatasetVersionDetail:
    """
    根据 hash 获取数据集版本

    Args:
        file_hash: 文件 hash
        session: 数据库会话

    Returns:
        数据集版本详情
    """
    repo = DatasetVersionRepository(session)
    version = repo.get_by_hash(file_hash)

    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset version with hash '{file_hash}' not found"
        )

    return DatasetVersionDetail(
        id=version.id,
        dataset_name=version.dataset_name,
        file_path=version.file_path,
        file_hash=version.file_hash,
        file_size_mb=version.file_size_mb,
        format=version.format,
        num_samples=version.num_samples,
        description=version.description,
        tags=version.tags,
        created_at=version.created_at.isoformat(),
        modified_at=version.modified_at,
    )


@router.get("/{file_hash}/lineage", response_model=DatasetLineageResponse)
async def trace_lineage(
    file_hash: str,
    session: Session = Depends(get_session),
) -> DatasetLineageResponse:
    """
    追溯数据集的血缘关系

    查找使用了该数据集版本的所有训练任务。

    Args:
        file_hash: 文件 hash
        session: 数据库会话

    Returns:
        血缘追踪结果
    """
    repo = DatasetVersionRepository(session)

    # 获取数据集版本
    version = repo.get_by_hash(file_hash)
    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset version with hash '{file_hash}' not found"
        )

    # 查找使用该版本的训练任务
    jobs = repo.find_jobs_using_version(file_hash)

    return DatasetLineageResponse(
        dataset_version=DatasetVersionDetail(
            id=version.id,
            dataset_name=version.dataset_name,
            file_path=version.file_path,
            file_hash=version.file_hash,
            file_size_mb=version.file_size_mb,
            format=version.format,
            num_samples=version.num_samples,
            description=version.description,
            tags=version.tags,
            created_at=version.created_at.isoformat(),
            modified_at=version.modified_at,
        ),
        used_by_jobs=[
            {
                "uuid": job.uuid,
                "name": job.name,
                "status": job.status,
                "algorithm": job.algorithm,
                "created_at": job.created_at.isoformat(),
            }
            for job in jobs
        ],
        num_jobs=len(jobs),
    )


@router.post("/compare", response_model=CompareVersionsResponse)
async def compare_versions(
    request: CompareVersionsRequest,
    session: Session = Depends(get_session),
) -> CompareVersionsResponse:
    """
    对比两个数据集版本

    检查两个版本是否相同，以及具体差异。

    Args:
        request: 对比请求
        session: 数据库会话

    Returns:
        对比结果
    """
    repo = DatasetVersionRepository(session)

    # 获取两个版本
    version_a = repo.get_by_hash(request.hash_a)
    version_b = repo.get_by_hash(request.hash_b)

    if not version_a or not version_b:
        raise HTTPException(
            status_code=404,
            detail="One or both dataset versions not found"
        )

    # 转换为快照格式
    snapshot_a = {
        "file_hash": version_a.file_hash,
        "file_size": version_a.file_size,
        "file_size_mb": version_a.file_size_mb,
        "num_samples": version_a.num_samples,
    }
    snapshot_b = {
        "file_hash": version_b.file_hash,
        "file_size": version_b.file_size,
        "file_size_mb": version_b.file_size_mb,
        "num_samples": version_b.num_samples,
    }

    # 对比
    result = compare_dataset_versions(snapshot_a, snapshot_b)

    return CompareVersionsResponse(**result)
