"""
Recipes API Router (Phase 2)

提供配方管理的 REST API 接口。
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from sqlmodel import Session

from ...core.recipes import (
    RecipeRegistry,
    TaskType,
    TrainingRecipe,
    apply_recipe_to_job_config,
    validate_recipe_config,
)
from ...core.database import get_session

router = APIRouter(prefix="/recipes", tags=["Training Recipes"])


# ============== Request/Response Models ==============

class RecipeSummary(BaseModel):
    """配方摘要（列表展示）"""
    id: str
    name: str
    description: str
    task_type: str
    algorithm: str
    tags: List[str] = []


class RecipeDetail(BaseModel):
    """配方详情"""
    id: str
    name: str
    description: str
    task_type: str
    recommended_algorithm: str
    default_config: Dict[str, Any]
    data_requirements: str
    model_size_hint: str
    min_gpus: int
    recommended_gpus: int
    tips: List[str]
    tags: List[str]
    author: str
    version: str


class ApplyRecipeRequest(BaseModel):
    """应用配方请求"""
    recipe_id: str = Field(..., description="配方 ID")
    overrides: Optional[Dict[str, Any]] = Field(None, description="自定义覆盖参数")
    model_size: Optional[str] = Field(None, description="模型大小提示（如 '7B', '70B'）")
    num_gpus: Optional[int] = Field(None, description="可用 GPU 数量")


class ApplyRecipeResponse(BaseModel):
    """应用配方响应"""
    recipe_id: str
    config: Dict[str, Any]
    warnings: List[str] = []


class ValidateConfigRequest(BaseModel):
    """验证配置请求"""
    recipe_id: str
    config: Dict[str, Any]


class ValidateConfigResponse(BaseModel):
    """验证配置响应"""
    valid: bool
    warnings: List[str] = []
    errors: List[str] = []


# ============== API Endpoints ==============

@router.get("", response_model=List[RecipeSummary])
async def list_recipes(
    task_type: Optional[str] = None,
    tag: Optional[str] = None,
) -> List[RecipeSummary]:
    """
    列出所有可用的训练配方

    可以按任务类型或标签筛选。

    Args:
        task_type: 任务类型筛选（sft, rlhf, dpo, grpo 等）
        tag: 标签筛选（beginner, advanced, lora 等）

    Returns:
        配方摘要列表
    """
    recipes = RecipeRegistry._recipes.values()

    # 按任务类型筛选
    if task_type:
        try:
            task_type_enum = TaskType(task_type)
            recipes = [r for r in recipes if r.task_type == task_type_enum]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_type: {task_type}"
            )

    # 按标签筛选
    if tag:
        recipes = [r for r in recipes if tag in r.tags]

    return [
        RecipeSummary(
            id=recipe.name,
            name=recipe.name.replace("_", " ").title(),
            description=recipe.description,
            task_type=recipe.task_type.value if hasattr(recipe.task_type, 'value') else recipe.task_type,
            algorithm=recipe.recommended_algorithm,
            tags=recipe.tags,
        )
        for recipe in recipes
    ]


@router.get("/{recipe_id}", response_model=RecipeDetail)
async def get_recipe(recipe_id: str) -> RecipeDetail:
    """
    获取指定配方的详细信息

    Args:
        recipe_id: 配方 ID

    Returns:
        配方详情
    """
    recipe = RecipeRegistry.get(recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=404,
            detail=f"Recipe '{recipe_id}' not found"
        )

    recipe_dict = recipe.to_dict()

    return RecipeDetail(
        id=recipe.name,
        name=recipe.name.replace("_", " ").title(),
        description=recipe.description,
        task_type=recipe_dict["task_type"],
        recommended_algorithm=recipe.recommended_algorithm,
        default_config=recipe.default_config,
        data_requirements=recipe.data_requirements,
        model_size_hint=recipe_dict["model_size_hint"],
        min_gpus=recipe.min_gpus,
        recommended_gpus=recipe.recommended_gpus,
        tips=recipe.tips,
        tags=recipe.tags,
        author=recipe.author,
        version=recipe.version,
    )


@router.post("/apply", response_model=ApplyRecipeResponse)
async def apply_recipe(request: ApplyRecipeRequest) -> ApplyRecipeResponse:
    """
    应用配方生成训练配置

    这是配方系统的核心功能：
    1. 获取配方的默认配置
    2. 根据模型大小和 GPU 数量自适应调整
    3. 应用用户自定义覆盖参数
    4. 验证配置并返回警告

    Args:
        request: 应用配方请求

    Returns:
        生成的配置和警告信息
    """
    # 获取配方
    recipe = RecipeRegistry.get(request.recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=404,
            detail=f"Recipe '{request.recipe_id}' not found"
        )

    # 获取自适应配置
    config = recipe.get_config(
        model_size=request.model_size,
        num_gpus=request.num_gpus,
    )

    # 应用用户覆盖
    if request.overrides:
        config = apply_recipe_to_job_config(recipe, request.overrides)
        # 如果有自适应参数，再次应用
        if request.model_size or request.num_gpus:
            adapted_config = recipe.get_config(
                model_size=request.model_size,
                num_gpus=request.num_gpus,
            )
            # 合并：用户覆盖优先
            for key, value in adapted_config.items():
                if key not in request.overrides:
                    config[key] = value

    # 验证配置
    warnings = validate_recipe_config(recipe, config)

    return ApplyRecipeResponse(
        recipe_id=request.recipe_id,
        config=config,
        warnings=warnings,
    )


@router.post("/validate", response_model=ValidateConfigResponse)
async def validate_config(request: ValidateConfigRequest) -> ValidateConfigResponse:
    """
    验证配置是否符合配方要求

    检查配置参数的合理性，返回警告和错误。

    Args:
        request: 验证请求

    Returns:
        验证结果
    """
    # 获取配方
    recipe = RecipeRegistry.get(request.recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=404,
            detail=f"Recipe '{request.recipe_id}' not found"
        )

    # 验证配置
    warnings = validate_recipe_config(recipe, request.config)

    # 检查严重错误
    errors = []
    required_params = ["learning_rate", "batch_size"]
    for param in required_params:
        if param not in request.config:
            errors.append(f"缺少必需参数: {param}")

    return ValidateConfigResponse(
        valid=len(errors) == 0,
        warnings=warnings,
        errors=errors,
    )


@router.get("/task-types", response_model=List[str])
async def list_task_types() -> List[str]:
    """
    列出所有支持的任务类型

    Returns:
        任务类型列表
    """
    return [task_type.value for task_type in TaskType]


@router.get("/tags", response_model=List[str])
async def list_tags() -> List[str]:
    """
    列出所有可用的标签

    Returns:
        标签列表（去重）
    """
    all_tags = set()
    for recipe in RecipeRegistry._recipes.values():
        all_tags.update(recipe.tags)
    return sorted(list(all_tags))
