"""
Model Merger for Model Surgery
Supports various merging methods: Linear, SLERP, TIES, DARE
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import os
import json
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MergeMethod(Enum):
    LINEAR = "linear"
    SLERP = "slerp"
    TIES = "ties"
    DARE = "dare"
    SWA = "swa"  # Stochastic Weight Averaging


@dataclass
class MergeConfig:
    """Configuration for model merging"""
    method: MergeMethod
    models: List[str]  # List of model paths
    weights: Optional[List[float]] = None
    output_path: Optional[str] = None

    # SLERP specific
    interpolation_t: float = 0.5

    # TIES specific
    density: float = 0.5  # Top-k% of parameters to keep

    # DARE specific
    drop_rate: float = 0.9  # Percentage of delta weights to drop

    # SWA specific
    start_step: Optional[int] = None  # Start averaging from this step

    def __post_init__(self):
        if self.weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(self.models)] * len(self.models)


@dataclass
class MergeResult:
    """Result of model merging operation"""
    success: bool
    output_path: Optional[str]
    method: str
    models_merged: List[str]
    weights_used: List[float]
    message: str
    metadata: Dict[str, Any]


class ModelMerger:
    """
    Model Merger supporting various merging algorithms
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def merge(self, config: MergeConfig) -> MergeResult:
        """
        Merge models according to configuration
        """
        if not TORCH_AVAILABLE:
            return MergeResult(
                success=False,
                output_path=None,
                method=config.method.value,
                models_merged=config.models,
                weights_used=config.weights,
                message="PyTorch not available",
                metadata={},
            )

        method_handlers = {
            MergeMethod.LINEAR: self._linear_merge,
            MergeMethod.SLERP: self._slerp_merge,
            MergeMethod.TIES: self._ties_merge,
            MergeMethod.DARE: self._dare_merge,
            MergeMethod.SWA: self._swa_merge,
        }

        handler = method_handlers.get(config.method)
        if handler is None:
            return MergeResult(
                success=False,
                output_path=None,
                method=config.method.value,
                models_merged=config.models,
                weights_used=config.weights,
                message=f"Unknown merge method: {config.method}",
                metadata={},
            )

        return handler(config)

    def _load_state_dict(self, model_path: str) -> Dict[str, Any]:
        """Load model state dict from path"""
        if os.path.isdir(model_path):
            # Try to find safetensors or pytorch files
            safetensors_file = os.path.join(model_path, "model.safetensors")
            pytorch_file = os.path.join(model_path, "pytorch_model.bin")

            if os.path.exists(safetensors_file):
                from safetensors.torch import load_file
                return load_file(safetensors_file, device=self.device)
            elif os.path.exists(pytorch_file):
                return torch.load(pytorch_file, map_location=self.device)
            else:
                # Try to load sharded model
                return self._load_sharded_model(model_path)
        else:
            # Direct file path
            if model_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(model_path, device=self.device)
            else:
                return torch.load(model_path, map_location=self.device)

    def _load_sharded_model(self, model_path: str) -> Dict[str, Any]:
        """Load sharded model from directory"""
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index = json.load(f)

            state_dict = {}
            shard_files = set(index["weight_map"].values())

            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                from safetensors.torch import load_file
                shard_dict = load_file(shard_path, device=self.device)
                state_dict.update(shard_dict)

            return state_dict
        else:
            raise ValueError(f"Cannot find model files in {model_path}")

    def _save_state_dict(
        self,
        state_dict: Dict[str, Any],
        output_path: str,
        use_safetensors: bool = True,
    ):
        """Save state dict to path"""
        os.makedirs(output_path, exist_ok=True)

        if use_safetensors:
            from safetensors.torch import save_file
            output_file = os.path.join(output_path, "model.safetensors")
            save_file(state_dict, output_file)
        else:
            output_file = os.path.join(output_path, "pytorch_model.bin")
            torch.save(state_dict, output_file)

        return output_file

    def _linear_merge(self, config: MergeConfig) -> MergeResult:
        """
        Linear interpolation merge: merged = sum(w_i * model_i)
        """
        try:
            # Normalize weights
            weights = np.array(config.weights)
            weights = weights / weights.sum()

            merged_state = None
            for i, model_path in enumerate(config.models):
                state_dict = self._load_state_dict(model_path)

                if merged_state is None:
                    merged_state = {
                        k: v.float() * weights[i] for k, v in state_dict.items()
                    }
                else:
                    for k, v in state_dict.items():
                        merged_state[k] += v.float() * weights[i]

            # Convert back to original dtype
            first_state = self._load_state_dict(config.models[0])
            for k in merged_state:
                merged_state[k] = merged_state[k].to(first_state[k].dtype)

            output_path = config.output_path or f"merged_linear_{len(config.models)}models"
            self._save_state_dict(merged_state, output_path)

            return MergeResult(
                success=True,
                output_path=output_path,
                method="linear",
                models_merged=config.models,
                weights_used=weights.tolist(),
                message="Linear merge completed successfully",
                metadata={"num_parameters": sum(p.numel() for p in merged_state.values())},
            )

        except Exception as e:
            return MergeResult(
                success=False,
                output_path=None,
                method="linear",
                models_merged=config.models,
                weights_used=config.weights,
                message=f"Linear merge failed: {str(e)}",
                metadata={},
            )

    def _slerp_merge(self, config: MergeConfig) -> MergeResult:
        """
        Spherical Linear Interpolation (SLERP) merge
        Better for merging two models while preserving their geometric properties
        """
        if len(config.models) != 2:
            return MergeResult(
                success=False,
                output_path=None,
                method="slerp",
                models_merged=config.models,
                weights_used=config.weights,
                message="SLERP requires exactly 2 models",
                metadata={},
            )

        try:
            state0 = self._load_state_dict(config.models[0])
            state1 = self._load_state_dict(config.models[1])
            t = config.interpolation_t

            merged_state = {}
            for key in state0.keys():
                v0 = state0[key].float().flatten()
                v1 = state1[key].float().flatten()

                merged_state[key] = self._slerp_vectors(v0, v1, t).reshape(
                    state0[key].shape
                ).to(state0[key].dtype)

            output_path = config.output_path or f"merged_slerp_t{t}"
            self._save_state_dict(merged_state, output_path)

            return MergeResult(
                success=True,
                output_path=output_path,
                method="slerp",
                models_merged=config.models,
                weights_used=[1 - t, t],
                message=f"SLERP merge completed with t={t}",
                metadata={"interpolation_t": t},
            )

        except Exception as e:
            return MergeResult(
                success=False,
                output_path=None,
                method="slerp",
                models_merged=config.models,
                weights_used=config.weights,
                message=f"SLERP merge failed: {str(e)}",
                metadata={},
            )

    def _slerp_vectors(
        self,
        v0: "torch.Tensor",
        v1: "torch.Tensor",
        t: float,
        dot_threshold: float = 0.9995,
    ) -> "torch.Tensor":
        """
        Spherical linear interpolation between two vectors
        """
        # Normalize vectors
        v0_norm = v0 / (torch.norm(v0) + 1e-8)
        v1_norm = v1 / (torch.norm(v1) + 1e-8)

        # Calculate dot product
        dot = torch.sum(v0_norm * v1_norm)

        # If vectors are too close, use linear interpolation
        if torch.abs(dot) > dot_threshold:
            return (1 - t) * v0 + t * v1

        # Calculate SLERP
        theta_0 = torch.acos(torch.clamp(dot, -1, 1))
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)

        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0

        return s0 * v0 + s1 * v1

    def _ties_merge(self, config: MergeConfig) -> MergeResult:
        """
        TIES (TrIm, Elect Sign & Merge) merge
        Good for merging multiple fine-tuned models
        """
        try:
            # Load base model (first model) and compute deltas
            base_state = self._load_state_dict(config.models[0])
            deltas = []

            for model_path in config.models[1:]:
                state = self._load_state_dict(model_path)
                delta = {
                    k: (state[k].float() - base_state[k].float())
                    for k in base_state.keys()
                }
                deltas.append(delta)

            # Step 1: Trim - keep only top-k% of each delta
            trimmed_deltas = []
            for delta in deltas:
                trimmed = {}
                for k, v in delta.items():
                    threshold = torch.quantile(
                        torch.abs(v.flatten()),
                        1 - config.density
                    )
                    mask = torch.abs(v) >= threshold
                    trimmed[k] = v * mask.float()
                trimmed_deltas.append(trimmed)

            # Step 2: Elect Sign - majority vote on sign
            merged_delta = {}
            for k in base_state.keys():
                stacked = torch.stack([d[k] for d in trimmed_deltas])
                signs = torch.sign(stacked)
                sign_sum = torch.sum(signs, dim=0)
                elected_sign = torch.sign(sign_sum)

                # Average magnitudes where signs agree
                magnitudes = torch.abs(stacked)
                sign_agreement = (signs == elected_sign.unsqueeze(0)).float()
                weighted_mag = (magnitudes * sign_agreement).sum(dim=0) / (
                    sign_agreement.sum(dim=0) + 1e-8
                )

                merged_delta[k] = elected_sign * weighted_mag

            # Step 3: Merge with base
            merged_state = {
                k: (base_state[k].float() + merged_delta[k]).to(base_state[k].dtype)
                for k in base_state.keys()
            }

            output_path = config.output_path or f"merged_ties_{len(config.models)}models"
            self._save_state_dict(merged_state, output_path)

            return MergeResult(
                success=True,
                output_path=output_path,
                method="ties",
                models_merged=config.models,
                weights_used=config.weights,
                message=f"TIES merge completed with density={config.density}",
                metadata={"density": config.density},
            )

        except Exception as e:
            return MergeResult(
                success=False,
                output_path=None,
                method="ties",
                models_merged=config.models,
                weights_used=config.weights,
                message=f"TIES merge failed: {str(e)}",
                metadata={},
            )

    def _dare_merge(self, config: MergeConfig) -> MergeResult:
        """
        DARE (Drop And REscale) merge
        Randomly drops delta weights and rescales, allowing more models to be merged
        """
        try:
            base_state = self._load_state_dict(config.models[0])
            merged_state = {k: v.clone().float() for k, v in base_state.items()}

            for i, model_path in enumerate(config.models[1:]):
                state = self._load_state_dict(model_path)
                weight = config.weights[i + 1] if len(config.weights) > i + 1 else 1.0

                for k in base_state.keys():
                    delta = state[k].float() - base_state[k].float()

                    # Randomly drop weights
                    mask = torch.rand_like(delta) > config.drop_rate
                    delta = delta * mask.float()

                    # Rescale to compensate for dropped weights
                    delta = delta / (1 - config.drop_rate + 1e-8)

                    merged_state[k] += delta * weight

            # Convert back to original dtype
            for k in merged_state:
                merged_state[k] = merged_state[k].to(base_state[k].dtype)

            output_path = config.output_path or f"merged_dare_{len(config.models)}models"
            self._save_state_dict(merged_state, output_path)

            return MergeResult(
                success=True,
                output_path=output_path,
                method="dare",
                models_merged=config.models,
                weights_used=config.weights,
                message=f"DARE merge completed with drop_rate={config.drop_rate}",
                metadata={"drop_rate": config.drop_rate},
            )

        except Exception as e:
            return MergeResult(
                success=False,
                output_path=None,
                method="dare",
                models_merged=config.models,
                weights_used=config.weights,
                message=f"DARE merge failed: {str(e)}",
                metadata={},
            )

    def _swa_merge(self, config: MergeConfig) -> MergeResult:
        """
        Stochastic Weight Averaging (SWA)
        Averages multiple checkpoints from the same training run
        """
        try:
            merged_state = None
            num_models = 0

            for model_path in config.models:
                # Filter by start_step if specified
                if config.start_step is not None:
                    # Try to extract step from path
                    step = self._extract_step_from_path(model_path)
                    if step is not None and step < config.start_step:
                        continue

                state = self._load_state_dict(model_path)
                num_models += 1

                if merged_state is None:
                    merged_state = {k: v.float() for k, v in state.items()}
                else:
                    for k, v in state.items():
                        merged_state[k] += v.float()

            if merged_state is None or num_models == 0:
                return MergeResult(
                    success=False,
                    output_path=None,
                    method="swa",
                    models_merged=config.models,
                    weights_used=config.weights,
                    message="No valid checkpoints found for SWA",
                    metadata={},
                )

            # Average
            for k in merged_state:
                merged_state[k] /= num_models

            # Get original dtype from first model
            first_state = self._load_state_dict(config.models[0])
            for k in merged_state:
                merged_state[k] = merged_state[k].to(first_state[k].dtype)

            output_path = config.output_path or f"merged_swa_{num_models}checkpoints"
            self._save_state_dict(merged_state, output_path)

            return MergeResult(
                success=True,
                output_path=output_path,
                method="swa",
                models_merged=config.models[:num_models],
                weights_used=[1.0 / num_models] * num_models,
                message=f"SWA completed: averaged {num_models} checkpoints",
                metadata={"num_checkpoints": num_models},
            )

        except Exception as e:
            return MergeResult(
                success=False,
                output_path=None,
                method="swa",
                models_merged=config.models,
                weights_used=config.weights,
                message=f"SWA failed: {str(e)}",
                metadata={},
            )

    def _extract_step_from_path(self, path: str) -> Optional[int]:
        """Extract step number from checkpoint path"""
        import re
        # Common patterns: ckpt-1000, step_1000, checkpoint-1000
        patterns = [
            r"ckpt-(\d+)",
            r"step[_-]?(\d+)",
            r"checkpoint[_-]?(\d+)",
            r"global_step(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, path)
            if match:
                return int(match.group(1))
        return None

    def scan_merge_ratios(
        self,
        model_a: str,
        model_b: str,
        method: MergeMethod = MergeMethod.SLERP,
        num_points: int = 9,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scan different merge ratios and return results
        Useful for finding optimal merge ratio
        """
        results = []
        ratios = np.linspace(0.1, 0.9, num_points)

        for t in ratios:
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"merged_t{t:.2f}")

            if method == MergeMethod.SLERP:
                config = MergeConfig(
                    method=method,
                    models=[model_a, model_b],
                    interpolation_t=t,
                    output_path=output_path,
                )
            else:
                config = MergeConfig(
                    method=method,
                    models=[model_a, model_b],
                    weights=[1 - t, t],
                    output_path=output_path,
                )

            result = self.merge(config)
            results.append({
                "ratio": t,
                "success": result.success,
                "output_path": result.output_path,
                "message": result.message,
            })

        return results


def merge_models(
    models: List[str],
    method: str = "slerp",
    weights: Optional[List[float]] = None,
    output_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for model merging
    """
    merger = ModelMerger()

    config = MergeConfig(
        method=MergeMethod(method),
        models=models,
        weights=weights,
        output_path=output_path,
        interpolation_t=kwargs.get("interpolation_t", 0.5),
        density=kwargs.get("density", 0.5),
        drop_rate=kwargs.get("drop_rate", 0.9),
        start_step=kwargs.get("start_step"),
    )

    result = merger.merge(config)

    return {
        "success": result.success,
        "output_path": result.output_path,
        "method": result.method,
        "models_merged": result.models_merged,
        "weights_used": result.weights_used,
        "message": result.message,
        "metadata": result.metadata,
    }
