"""
Command Utilities - Safe command construction

提供安全的命令构造工具，防止命令注入攻击。
"""

import shlex
import re
from pathlib import Path
from typing import List, Union


def quote_argument(arg: Union[str, int, Path]) -> str:
    """
    安全地引用命令行参数，防止命令注入。

    Args:
        arg: 要引用的参数（字符串、整数或路径）

    Returns:
        安全引用的字符串
    """
    return shlex.quote(str(arg))


def build_command(*args: Union[str, int, Path]) -> str:
    """
    安全地构造命令字符串。

    每个参数都会被自动引用以防止注入。

    Example:
        >>> build_command('ls', '-la', '/tmp/file with spaces.txt')
        "ls -la '/tmp/file with spaces.txt'"

    Args:
        *args: 命令和参数

    Returns:
        安全的命令字符串
    """
    return ' '.join(quote_argument(arg) for arg in args)


def validate_path(path: str, allow_special_chars: bool = False) -> bool:
    """
    验证路径是否安全。

    检查路径是否包含危险的字符或模式。

    Args:
        path: 要验证的路径
        allow_special_chars: 是否允许特殊字符（如 ~, *, ?）

    Returns:
        是否为安全路径
    """
    if not path:
        return False

    # 检查危险模式
    dangerous_patterns = [
        r'\|',      # 管道
        r';',       # 命令分隔符
        r'&&',      # 命令连接
        r'\|\|',    # 命令连接
        r'\$\(',    # 命令替换
        r'`',       # 命令替换
        r'\.\./',   # 路径遍历（仅在开头或中间）
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, path):
            return False

    # 如果不允许特殊字符，检查通配符
    if not allow_special_chars:
        if any(c in path for c in ['*', '?', '[', ']']):
            return False

    return True


def validate_identifier(identifier: str) -> bool:
    """
    验证标识符（如作业 ID、文件名）是否安全。

    仅允许字母、数字、下划线、连字符和点。

    Args:
        identifier: 要验证的标识符

    Returns:
        是否为安全标识符
    """
    if not identifier:
        return False

    # 仅允许安全字符
    return bool(re.match(r'^[a-zA-Z0-9._-]+$', identifier))


def validate_integer(value: Union[str, int], min_val: int = None, max_val: int = None) -> bool:
    """
    验证整数值是否在安全范围内。

    Args:
        value: 要验证的值
        min_val: 最小值（可选）
        max_val: 最大值（可选）

    Returns:
        是否为有效整数
    """
    try:
        int_val = int(value)

        if min_val is not None and int_val < min_val:
            return False

        if max_val is not None and int_val > max_val:
            return False

        return True
    except (ValueError, TypeError):
        return False


# 预定义的安全命令模板
class SafeCommands:
    """常用安全命令的模板"""

    @staticmethod
    def ps_check(pid: int) -> str:
        """检查进程是否存在"""
        if not validate_integer(pid, min_val=1):
            raise ValueError(f"Invalid PID: {pid}")
        return f"ps -p {pid} -o pid= 2>/dev/null"

    @staticmethod
    def tail_file(file_path: str, lines: int = 100) -> str:
        """读取文件尾部"""
        if not validate_path(file_path):
            raise ValueError(f"Invalid path: {file_path}")
        if not validate_integer(lines, min_val=1, max_val=10000):
            raise ValueError(f"Invalid line count: {lines}")

        return build_command('tail', '-n', lines, file_path)

    @staticmethod
    def tail_follow(file_path: str) -> str:
        """持续跟踪文件"""
        if not validate_path(file_path):
            raise ValueError(f"Invalid path: {file_path}")

        return build_command('tail', '-f', file_path)

    @staticmethod
    def mkdir(dir_path: str) -> str:
        """创建目录"""
        if not validate_path(dir_path, allow_special_chars=True):
            raise ValueError(f"Invalid directory path: {dir_path}")

        return build_command('mkdir', '-p', dir_path)

    @staticmethod
    def rm_file(file_path: str) -> str:
        """删除文件"""
        if not validate_path(file_path):
            raise ValueError(f"Invalid file path: {file_path}")

        return build_command('rm', '-f', file_path)

    @staticmethod
    def ls_file(file_path: str) -> str:
        """列出文件信息"""
        if not validate_path(file_path, allow_special_chars=True):
            raise ValueError(f"Invalid file path: {file_path}")

        return build_command('ls', '-la', file_path) + ' 2>/dev/null'

    @staticmethod
    def echo_expand_path(path: str) -> str:
        """展开路径（如 ~ 扩展）"""
        if not validate_path(path, allow_special_chars=True):
            raise ValueError(f"Invalid path: {path}")

        return build_command('echo', path)


if __name__ == '__main__':
    # 测试
    print("=== 命令安全工具测试 ===\n")

    # 测试安全命令构造
    cmd = build_command('ls', '-la', '/tmp/file with spaces.txt')
    print(f"安全命令: {cmd}\n")

    # 测试路径验证
    safe_paths = [
        "/home/user/data.txt",
        "~/workspace/project",
        "./local/file.json"
    ]
    dangerous_paths = [
        "/home/user; rm -rf /",
        "file.txt | malicious",
        "$(whoami)/file",
        "`cat /etc/passwd`"
    ]

    print("安全路径:")
    for path in safe_paths:
        print(f"  {path}: {validate_path(path, allow_special_chars=True)}")

    print("\n危险路径:")
    for path in dangerous_paths:
        print(f"  {path}: {validate_path(path, allow_special_chars=True)}")

    # 测试预定义命令
    print("\n=== 预定义安全命令 ===")
    print(f"tail: {SafeCommands.tail_file('/var/log/app.log', 50)}")
    print(f"mkdir: {SafeCommands.mkdir('~/work/project')}")
    print(f"ps: {SafeCommands.ps_check(1234)}")
