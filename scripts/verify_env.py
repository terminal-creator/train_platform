#!/usr/bin/env python3
"""
Training Platform - Environment Verification Script
环境验证脚本

用途：验证环境是否正确安装并符合要求
使用方法：python scripts/verify_env.py --mode [manager|training]
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util


# 颜色输出
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def log_info(msg: str):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")


def log_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def log_section(title: str):
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BLUE}{title:^60}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}\n")


class EnvironmentVerifier:
    def __init__(self, mode: str, project_root: Path):
        self.mode = mode  # manager or training
        self.project_root = project_root
        self.env_dir = project_root / "environments"
        self.version_file = self.env_dir / "version.json"

        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed_checks: List[str] = []

        # 加载版本要求
        self.version_spec = self._load_version_spec()

    def _load_version_spec(self) -> Dict:
        """加载 version.json"""
        if not self.version_file.exists():
            log_warn(f"未找到 version.json: {self.version_file}")
            return {}

        with open(self.version_file) as f:
            return json.load(f)

    def check_python_version(self) -> bool:
        """检查 Python 版本"""
        log_section("Python 版本检查")

        version_info = sys.version_info
        version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

        log_info(f"当前 Python 版本: {version_str}")
        log_info(f"版本信息: {sys.version}")

        # 检查版本要求
        if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 9):
            self.errors.append(f"Python 版本过低: {version_str}，需要 >= 3.9")
            return False

        if version_info.major == 3 and version_info.minor >= 12:
            self.warnings.append(f"Python 版本较新: {version_str}，推荐 3.9-3.11")

        self.passed_checks.append(f"Python 版本: {version_str}")
        return True

    def check_package(self, package_name: str, expected_version: Optional[str] = None) -> bool:
        """检查包是否已安装及版本"""
        try:
            # 尝试导入包
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                return False

            # 获取版本
            module = importlib.import_module(package_name)
            actual_version = getattr(module, '__version__', 'unknown')

            # 显示信息
            if expected_version:
                if actual_version == expected_version:
                    log_info(f"✓ {package_name}: {actual_version}")
                    self.passed_checks.append(f"{package_name}: {actual_version}")
                else:
                    log_warn(f"⚠ {package_name}: {actual_version} (期望: {expected_version})")
                    self.warnings.append(f"{package_name} 版本不匹配: {actual_version} != {expected_version}")
            else:
                log_info(f"✓ {package_name}: {actual_version}")
                self.passed_checks.append(f"{package_name}: {actual_version}")

            return True

        except ImportError as e:
            log_error(f"✗ {package_name}: 未安装")
            self.errors.append(f"{package_name} 未安装: {e}")
            return False
        except Exception as e:
            log_error(f"✗ {package_name}: 检查失败 - {e}")
            self.errors.append(f"{package_name} 检查失败: {e}")
            return False

    def check_dependencies(self) -> bool:
        """检查关键依赖"""
        log_section("依赖包检查")

        # 基础依赖
        basic_packages = {
            'fastapi': self.version_spec.get('key_dependencies', {}).get('fastapi'),
            'pydantic': None,
            'sqlmodel': None,
            'sqlalchemy': self.version_spec.get('key_dependencies', {}).get('sqlalchemy'),
            'paramiko': None,
            'numpy': None,
        }

        # 训练相关依赖
        if self.mode == 'training':
            training_packages = {
                'torch': self.version_spec.get('key_dependencies', {}).get('pytorch'),
                'transformers': self.version_spec.get('key_dependencies', {}).get('transformers'),
                'datasets': None,
                'hydra': None,
            }
            basic_packages.update(training_packages)

        all_passed = True
        for package, expected_version in basic_packages.items():
            if not self.check_package(package, expected_version):
                all_passed = False

        return all_passed

    def check_cuda_gpu(self) -> bool:
        """检查 CUDA 和 GPU（仅训练模式）"""
        if self.mode != 'training':
            return True

        log_section("CUDA / GPU 检查")

        try:
            import torch

            # 检查 CUDA 可用性
            cuda_available = torch.cuda.is_available()
            log_info(f"CUDA 可用: {cuda_available}")

            if not cuda_available:
                self.warnings.append("CUDA 不可用，将使用 CPU 模式")
                return True

            # 检查 CUDA 版本
            cuda_version = torch.version.cuda
            log_info(f"CUDA 版本: {cuda_version}")

            # 检查 GPU 数量
            gpu_count = torch.cuda.device_count()
            log_info(f"GPU 数量: {gpu_count}")

            # 检查每个 GPU
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                total_mem = gpu_props.total_memory / 1024 ** 3  # GB
                log_info(f"  GPU {i}: {gpu_name} ({total_mem:.1f} GB)")

            self.passed_checks.append(f"CUDA: {cuda_version}, {gpu_count} GPU(s)")
            return True

        except ImportError:
            log_error("PyTorch 未安装，无法检查 CUDA")
            self.errors.append("PyTorch 未安装")
            return False
        except Exception as e:
            log_error(f"CUDA 检查失败: {e}")
            self.errors.append(f"CUDA 检查失败: {e}")
            return False

    def check_verl(self) -> bool:
        """检查 verl 安装"""
        log_section("verl 检查")

        # 检查 verl 目录
        verl_dir = self.env_dir / "verl"
        if not verl_dir.exists():
            log_error("verl 目录不存在")
            self.errors.append("verl submodule 未初始化")
            return False

        log_info(f"✓ verl 目录存在: {verl_dir}")

        # 检查 verl 是否可导入
        try:
            import verl
            verl_version = getattr(verl, '__version__', 'unknown')
            log_info(f"✓ verl 可导入，版本: {verl_version}")

            # 检查 commit
            expected_commit = self.version_spec.get('verl', {}).get('commit', '')
            if expected_commit:
                log_info(f"期望 commit: {expected_commit[:8]}")

            self.passed_checks.append(f"verl: {verl_version}")
            return True

        except ImportError as e:
            log_warn(f"verl 无法导入: {e}")
            log_warn("这可能是正常的（verl 依赖完整的训练环境）")
            self.warnings.append("verl 未安装或无法导入")
            return True  # 不作为错误
        except Exception as e:
            log_error(f"verl 检查失败: {e}")
            self.errors.append(f"verl 检查失败: {e}")
            return False

    def check_environment_consistency(self) -> bool:
        """检查环境一致性"""
        log_section("环境一致性检查")

        # 检查虚拟环境
        in_venv = sys.prefix != sys.base_prefix
        log_info(f"虚拟环境: {in_venv}")
        if not in_venv:
            self.warnings.append("未在虚拟环境中运行")

        # 检查 pip 包数量
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
                capture_output=True,
                text=True
            )
            installed_packages = len(result.stdout.strip().split('\n'))
            log_info(f"已安装包数量: {installed_packages}")
            self.passed_checks.append(f"已安装 {installed_packages} 个包")
        except Exception as e:
            log_warn(f"无法统计安装包数量: {e}")

        return True

    def run_all_checks(self) -> bool:
        """运行所有检查"""
        print(f"\n{'=' * 60}")
        print(f"训练平台环境验证 - {self.mode.upper()} 模式")
        print(f"{'=' * 60}\n")

        # 执行各项检查
        checks = [
            self.check_python_version(),
            self.check_dependencies(),
            self.check_cuda_gpu(),
            self.check_verl(),
            self.check_environment_consistency(),
        ]

        # 生成报告
        self.generate_report()

        return all(checks) and len(self.errors) == 0

    def generate_report(self):
        """生成验证报告"""
        log_section("验证报告")

        # 通过的检查
        if self.passed_checks:
            log_info(f"通过检查: {len(self.passed_checks)} 项")
            for check in self.passed_checks[:5]:  # 只显示前5项
                print(f"  ✓ {check}")
            if len(self.passed_checks) > 5:
                print(f"  ... 还有 {len(self.passed_checks) - 5} 项通过")

        # 警告
        if self.warnings:
            print()
            log_warn(f"警告: {len(self.warnings)} 项")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        # 错误
        if self.errors:
            print()
            log_error(f"错误: {len(self.errors)} 项")
            for error in self.errors:
                print(f"  ✗ {error}")

        # 总结
        print()
        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                log_info(f"{Colors.GREEN}✓ 环境验证通过！{Colors.NC}")
            else:
                log_warn(f"⚠ 环境验证通过，但有 {len(self.warnings)} 个警告")
        else:
            log_error(f"✗ 环境验证失败，有 {len(self.errors)} 个错误")
        print()


def main():
    parser = argparse.ArgumentParser(description='验证训练平台环境')
    parser.add_argument(
        '--mode',
        choices=['manager', 'training'],
        default='manager',
        help='节点类型：manager (管理节点) 或 training (训练节点)'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path(__file__).parent.parent,
        help='项目根目录路径'
    )

    args = parser.parse_args()

    # 执行验证
    verifier = EnvironmentVerifier(args.mode, args.project_root)
    success = verifier.run_all_checks()

    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
