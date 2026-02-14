"""
Demo计算配置数据
"""
from typing import Dict, List, Any


DEMO_GPU_TYPES = [
    {"name": "A100-80G", "memory_gb": 80, "tflops": 312, "recommended": True},
    {"name": "A100-40G", "memory_gb": 40, "tflops": 312, "recommended": False},
    {"name": "H100-80G", "memory_gb": 80, "tflops": 989, "recommended": True},
    {"name": "H100-SXM", "memory_gb": 80, "tflops": 989, "recommended": False},
    {"name": "A800-80G", "memory_gb": 80, "tflops": 312, "recommended": False},
    {"name": "H800-80G", "memory_gb": 80, "tflops": 989, "recommended": False},
    {"name": "RTX4090", "memory_gb": 24, "tflops": 82.6, "recommended": False},
    {"name": "L40S", "memory_gb": 48, "tflops": 91.6, "recommended": False},
]

DEMO_MODEL_SIZES = [
    {"name": "0.5B", "params_billion": 0.5, "layers": 24},
    {"name": "1.5B", "params_billion": 1.5, "layers": 28},
    {"name": "3B", "params_billion": 3, "layers": 32},
    {"name": "7B", "params_billion": 7, "layers": 32},
    {"name": "14B", "params_billion": 14, "layers": 40},
    {"name": "32B", "params_billion": 32, "layers": 60},
    {"name": "72B", "params_billion": 72, "layers": 80},
]


def get_demo_compute_result(
    model_size: str = "7B",
    gpu_type: str = "A100-80G",
    gpu_count: int = 8,
    training_type: str = "grpo",
    sequence_length: int = 4096,
    use_lora: bool = False,
) -> Dict[str, Any]:
    model_info = next((m for m in DEMO_MODEL_SIZES if m["name"] == model_size), DEMO_MODEL_SIZES[3])
    gpu_info = next((g for g in DEMO_GPU_TYPES if g["name"] == gpu_type), DEMO_GPU_TYPES[0])

    params_b = model_info["params_billion"]
    gpu_memory = gpu_info["memory_gb"]

    model_memory = params_b * 2
    optimizer_memory = params_b * 8 if not use_lora else params_b * 0.5
    gradient_memory = params_b * 2
    activation_memory = min(params_b * 2, 8)

    total_memory_per_gpu = (model_memory + optimizer_memory + gradient_memory + activation_memory) / gpu_count
    if total_memory_per_gpu > gpu_memory * 0.9:
        zero_stage = 3
    elif total_memory_per_gpu > gpu_memory * 0.7:
        zero_stage = 2
    else:
        zero_stage = 1

    available_memory = gpu_memory * 0.85 - total_memory_per_gpu
    micro_batch_size = max(1, int(available_memory / (sequence_length * 0.001)))
    micro_batch_size = min(micro_batch_size, 8)

    gradient_accumulation = max(1, 256 // (micro_batch_size * gpu_count))
    global_batch_size = micro_batch_size * gpu_count * gradient_accumulation

    if training_type == "sft":
        learning_rate = 1e-5
    elif training_type in ["grpo", "ppo"]:
        learning_rate = 5e-7
    elif training_type == "dpo":
        learning_rate = 5e-7
    else:
        learning_rate = 1e-5

    config = {
        "actor": {
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "gradient_accumulation_steps": gradient_accumulation,
            "micro_batch_size": micro_batch_size,
            "max_grad_norm": 1.0,
        },
        "model": {
            "model_path": f"/models/Qwen2.5-{model_size}-Instruct",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
        },
        "data": {
            "max_seq_length": sequence_length,
            "train_batch_size": global_batch_size,
        },
        "deepspeed": {
            "zero_stage": zero_stage,
            "offload_optimizer": zero_stage == 3,
            "offload_param": zero_stage == 3,
        },
    }

    if training_type in ["grpo", "ppo"]:
        config["critic"] = {"learning_rate": learning_rate * 2}
        config["rollout"] = {"num_gpus": gpu_count // 2, "tensor_parallel_size": 2 if params_b >= 14 else 1}
        config["ref"] = {"num_gpus": gpu_count // 4}
        config["ppo"] = {"kl_coef": 0.04, "clip_range": 0.2, "value_loss_coef": 0.5, "entropy_coef": 0.01, "ppo_epochs": 4}
    elif training_type == "dpo":
        config["dpo"] = {"beta": 0.1, "label_smoothing": 0.0, "loss_type": "sigmoid"}

    if use_lora:
        config["lora"] = {"r": 64, "lora_alpha": 128, "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}

    yaml_content = _generate_yaml(config, training_type)

    memory_estimate = {
        "model_weights_gb": round(model_memory, 1),
        "optimizer_states_gb": round(optimizer_memory, 1),
        "gradients_gb": round(gradient_memory, 1),
        "activations_gb": round(activation_memory, 1),
        "total_gb": round(model_memory + optimizer_memory + gradient_memory + activation_memory, 1),
        "per_gpu_gb": round(total_memory_per_gpu, 1),
        "available_gpu_memory_gb": gpu_memory,
        "memory_utilization": round(total_memory_per_gpu / gpu_memory * 100, 1),
    }

    summary = {
        "model_size": model_size,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "training_type": training_type,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "gradient_accumulation": gradient_accumulation,
        "zero_stage": zero_stage,
        "estimated_memory_per_gpu": f"{memory_estimate['per_gpu_gb']:.1f}GB / {gpu_memory}GB",
        "estimated_throughput": f"{int(global_batch_size * 1000 / (params_b * 2))} tokens/sec",
        "use_lora": use_lora,
        "recommendations": _generate_recommendations(params_b, gpu_memory, gpu_count, zero_stage, training_type),
    }

    return {
        "config": config,
        "yaml": yaml_content,
        "memory_estimate": memory_estimate,
        "zero_stage": zero_stage,
        "summary": summary,
    }


def _generate_yaml(config: Dict, training_type: str) -> str:
    lines = ["# Auto-generated VERL configuration", f"# Training type: {training_type.upper()}", ""]

    def add_section(name: str, data: Dict, indent: int = 0):
        prefix = "  " * indent
        lines.append(f"{prefix}{name}:")
        for key, value in data.items():
            if isinstance(value, dict):
                add_section(key, value, indent + 1)
            elif isinstance(value, bool):
                lines.append(f"{prefix}  {key}: {'true' if value else 'false'}")
            elif isinstance(value, float):
                if value < 0.001:
                    lines.append(f"{prefix}  {key}: {value:.2e}")
                else:
                    lines.append(f"{prefix}  {key}: {value}")
            elif isinstance(value, list):
                lines.append(f"{prefix}  {key}:")
                for item in value:
                    lines.append(f"{prefix}    - {item}")
            else:
                lines.append(f"{prefix}  {key}: {value}")

    for section_name, section_data in config.items():
        if isinstance(section_data, dict):
            add_section(section_name, section_data)
            lines.append("")

    return "\n".join(lines)


def _generate_recommendations(params_b: float, gpu_memory: float, gpu_count: int, zero_stage: int, training_type: str) -> List[str]:
    recommendations = []
    if zero_stage >= 2:
        recommendations.append(f"使用ZeRO Stage {zero_stage}以优化内存使用")
    if params_b >= 14 and gpu_count >= 4:
        recommendations.append("建议启用Tensor Parallelism以加速大模型训练")
    if training_type in ["grpo", "ppo"]:
        recommendations.append("GRPO/PPO训练建议使用较小的KL系数(0.02-0.05)保持稳定性")
    if params_b >= 7:
        recommendations.append("建议使用BF16混合精度训练以平衡速度和精度")
    if gpu_memory < 80:
        recommendations.append("内存较小时建议使用Gradient Checkpointing")
    return recommendations


def get_gpu_types() -> List[Dict]:
    return DEMO_GPU_TYPES


def get_model_sizes() -> List[Dict]:
    return DEMO_MODEL_SIZES
