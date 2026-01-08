"""
Cryptography utilities for secure storage of sensitive data.

提供敏感数据（如 SSH 密码）的加密存储功能。
"""

import os
import base64
import logging
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

# 环境变量中的密钥名称
PLATFORM_SECRET_KEY_ENV = "PLATFORM_SECRET_KEY"

# 默认的盐（用于密钥派生）
DEFAULT_SALT = b'training_platform_salt_v1'


class CryptoError(Exception):
    """加密/解密错误"""
    pass


class SecretManager:
    """
    密钥管理器

    使用 Fernet 对称加密来保护敏感数据（如 SSH 密码）。
    密钥从环境变量读取，如果不存在则自动生成（仅开发环境）。
    """

    def __init__(self):
        self._fernet: Optional[Fernet] = None
        self._init_fernet()

    def _init_fernet(self):
        """初始化 Fernet 加密器"""
        try:
            # 1. 尝试从环境变量读取密钥
            secret_key = os.getenv(PLATFORM_SECRET_KEY_ENV)

            if secret_key:
                logger.info("使用环境变量中的加密密钥")
                key_bytes = secret_key.encode()
            else:
                # 2. 生产环境必须设置环境变量
                logger.warning(
                    f"未设置环境变量 {PLATFORM_SECRET_KEY_ENV}，"
                    "使用默认密钥（仅适用于开发环境！）"
                )

                # 使用默认密钥（基于机器信息生成）
                key_bytes = self._generate_default_key()

            # 派生 Fernet 密钥（必须是 32 字节的 URL-safe base64 编码）
            fernet_key = self._derive_fernet_key(key_bytes)
            self._fernet = Fernet(fernet_key)

        except Exception as e:
            raise CryptoError(f"初始化加密器失败: {e}")

    def _generate_default_key(self) -> bytes:
        """生成默认密钥（开发环境）"""
        # 警告：这个密钥不安全，仅用于开发！
        import socket
        hostname = socket.gethostname()
        default_key = f"dev_key_{hostname}".encode()
        return default_key

    def _derive_fernet_key(self, password: bytes) -> bytes:
        """
        从密码派生 Fernet 密钥

        使用 PBKDF2 从任意长度的密码派生出 32 字节的密钥。
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=DEFAULT_SALT,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

    def encrypt(self, plaintext: str) -> str:
        """
        加密字符串

        Args:
            plaintext: 明文字符串

        Returns:
            加密后的字符串（base64 编码）
        """
        if not plaintext:
            return ""

        try:
            encrypted_bytes = self._fernet.encrypt(plaintext.encode())
            return encrypted_bytes.decode()
        except Exception as e:
            raise CryptoError(f"加密失败: {e}")

    def decrypt(self, ciphertext: str) -> str:
        """
        解密字符串

        Args:
            ciphertext: 加密的字符串（base64 编码）

        Returns:
            解密后的明文字符串
        """
        if not ciphertext:
            return ""

        try:
            decrypted_bytes = self._fernet.decrypt(ciphertext.encode())
            return decrypted_bytes.decode()
        except Exception as e:
            raise CryptoError(f"解密失败: {e}")

    def is_encrypted(self, text: str) -> bool:
        """
        判断字符串是否已加密

        通过尝试解密来判断。
        """
        if not text:
            return False

        try:
            self.decrypt(text)
            return True
        except:
            return False


# 全局单例
_secret_manager: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    """获取全局密钥管理器"""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager


def encrypt_password(password: str) -> str:
    """
    加密密码

    这是一个便捷函数，用于加密 SSH 密码等敏感信息。
    """
    if not password:
        return ""
    return get_secret_manager().encrypt(password)


def decrypt_password(encrypted_password: str) -> str:
    """
    解密密码

    这是一个便捷函数，用于解密 SSH 密码等敏感信息。
    """
    if not encrypted_password:
        return ""
    return get_secret_manager().decrypt(encrypted_password)


def generate_secret_key() -> str:
    """
    生成一个安全的随机密钥

    用于生产环境设置 PLATFORM_SECRET_KEY 环境变量。

    Returns:
        32 字节的随机密钥（hex 编码）
    """
    return os.urandom(32).hex()


# 用于生产环境的密钥生成脚本
if __name__ == "__main__":
    print("=" * 60)
    print("训练平台密钥生成器")
    print("=" * 60)
    print()
    print("生成新的加密密钥...")
    key = generate_secret_key()
    print()
    print("请将以下密钥设置为环境变量（保密！）：")
    print()
    print(f"export {PLATFORM_SECRET_KEY_ENV}='{key}'")
    print()
    print("或者添加到 ~/.bashrc 或 ~/.zshrc：")
    print(f"echo \"export {PLATFORM_SECRET_KEY_ENV}='{key}'\" >> ~/.bashrc")
    print()
    print("=" * 60)
    print("⚠️  重要：请妥善保管此密钥，不要泄露！")
    print("=" * 60)
