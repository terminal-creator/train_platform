#!/usr/bin/env python3
"""
数据库迁移脚本 - Phase 1.2

为 TrainingMetric 表添加新字段：
- reward_max, reward_min: 奖励的最大/最小值
- kl_divergence_max: KL 散度最大值（用于异常检测）
- grad_norm_actor, grad_norm_critic: 梯度范数
- tokens_per_second, step_time: 性能指标
- gpu_memory_allocated_gib: GPU 内存使用
- has_anomaly, anomaly_type, anomaly_message: 异常检测标记

为什么这么设计：
- SQLite 不支持删除列，但支持添加列
- 新字段都设置为可选（允许 NULL）
- 使用 ALTER TABLE 逐个添加列
- 如果列已存在，会跳过（幂等性）
"""

import os
import sys
import sqlite3
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_db_path() -> Path:
    """获取数据库文件路径"""
    db_path = project_root / "training_platform.db"
    if not db_path.exists():
        print(f"❌ 数据库文件不存在: {db_path}")
        print("   请先启动平台创建数据库")
        sys.exit(1)
    return db_path


def check_column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    """检查列是否已存在"""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def add_column_if_not_exists(
    cursor: sqlite3.Cursor,
    table: str,
    column: str,
    column_type: str,
    default_value: str = "NULL"
) -> bool:
    """如果列不存在，则添加"""
    if check_column_exists(cursor, table, column):
        print(f"  ⏭️  列已存在: {column}")
        return False

    try:
        sql = f"ALTER TABLE {table} ADD COLUMN {column} {column_type} DEFAULT {default_value}"
        cursor.execute(sql)
        print(f"  ✓ 添加列: {column} ({column_type})")
        return True
    except sqlite3.Error as e:
        print(f"  ❌ 添加列失败 {column}: {e}")
        return False


def migrate_training_metrics_table(cursor: sqlite3.Cursor):
    """迁移 training_metrics 表"""
    print("\n" + "=" * 60)
    print("迁移 training_metrics 表")
    print("=" * 60)

    table_name = "training_metrics"

    # 检查表是否存在
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    if not cursor.fetchone():
        print(f"❌ 表不存在: {table_name}")
        print("   请先启动平台创建数据库")
        return False

    # 需要添加的新列
    new_columns = [
        # 奖励指标
        ("reward_max", "REAL", "NULL"),
        ("reward_min", "REAL", "NULL"),

        # KL 散度指标
        ("kl_divergence_max", "REAL", "NULL"),

        # 梯度指标
        ("grad_norm_actor", "REAL", "NULL"),
        ("grad_norm_critic", "REAL", "NULL"),

        # 性能指标
        ("tokens_per_second", "REAL", "NULL"),
        ("step_time", "REAL", "NULL"),
        ("gpu_memory_allocated_gib", "REAL", "NULL"),

        # 异常检测
        ("has_anomaly", "BOOLEAN", "0"),  # 默认 False
        ("anomaly_type", "VARCHAR", "NULL"),
        ("anomaly_message", "VARCHAR", "NULL"),
    ]

    added_count = 0
    for column, column_type, default_value in new_columns:
        if add_column_if_not_exists(cursor, table_name, column, column_type, default_value):
            added_count += 1

    print(f"\n✓ 共添加 {added_count} 个新列")
    return True


def verify_schema(cursor: sqlite3.Cursor):
    """验证迁移后的表结构"""
    print("\n" + "=" * 60)
    print("验证表结构")
    print("=" * 60)

    cursor.execute("PRAGMA table_info(training_metrics)")
    columns = cursor.fetchall()

    print(f"\ntraining_metrics 表共有 {len(columns)} 列:")
    print()
    print(f"{'列名':<30} {'类型':<15} {'可空':<10}")
    print("-" * 55)

    for col in columns:
        col_id, name, col_type, not_null, default_val, pk = col
        nullable = "NO" if not_null else "YES"
        print(f"{name:<30} {col_type:<15} {nullable:<10}")

    # 检查所有必需的列是否存在
    required_columns = [
        "reward_max", "reward_min", "kl_divergence_max",
        "grad_norm_actor", "grad_norm_critic",
        "tokens_per_second", "step_time", "gpu_memory_allocated_gib",
        "has_anomaly", "anomaly_type", "anomaly_message"
    ]

    existing_columns = [col[1] for col in columns]
    missing_columns = [col for col in required_columns if col not in existing_columns]

    if missing_columns:
        print(f"\n❌ 缺少列: {', '.join(missing_columns)}")
        return False
    else:
        print(f"\n✓ 所有必需列都已存在")
        return True


def main():
    """主函数"""
    print("=" * 60)
    print("数据库迁移脚本 - Phase 1.2")
    print("=" * 60)

    # 获取数据库路径
    db_path = get_db_path()
    print(f"\n数据库文件: {db_path}")

    # 备份数据库（可选但推荐）
    backup_path = db_path.with_suffix(".db.backup")
    if backup_path.exists():
        print(f"⚠️  备份文件已存在: {backup_path}")
    else:
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✓ 已备份数据库到: {backup_path}")

    # 连接数据库
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 执行迁移
        success = migrate_training_metrics_table(cursor)

        if success:
            # 提交更改
            conn.commit()
            print("\n✓ 数据库迁移完成")

            # 验证表结构
            verify_schema(cursor)
        else:
            conn.rollback()
            print("\n❌ 数据库迁移失败")
            return 1

    except Exception as e:
        print(f"\n❌ 迁移出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if conn:
            conn.close()

    print("\n" + "=" * 60)
    print("迁移完成！")
    print("=" * 60)
    print("\n现在可以运行测试脚本: python tests/test_phase1_2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
