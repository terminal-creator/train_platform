"""
Database Migration Script for Dataset Statistics

Adds pre-computed statistics and quality fields to TrainingDataset:
1. statistics - JSON field for cached dataset statistics
2. quality_stats - JSON field for cached quality check results

Usage:
    python -m training_platform.core.migrate_dataset_stats
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlmodel import Session, text
from training_platform.core.database import engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_sqlite():
    """Migration for SQLite database"""
    logger.info("Starting SQLite migration for dataset statistics...")

    with Session(engine) as session:
        try:
            # Check if columns already exist
            result = session.exec(text("PRAGMA table_info(training_datasets)"))
            columns = [row[1] for row in result.fetchall()]

            # Add statistics column if it doesn't exist
            if "statistics" not in columns:
                logger.info("Adding statistics column to training_datasets...")
                session.exec(
                    text("ALTER TABLE training_datasets ADD COLUMN statistics TEXT DEFAULT '{}'")
                )
                logger.info("✓ Added statistics column")
            else:
                logger.info("statistics column already exists")

            # Add quality_stats column if it doesn't exist
            if "quality_stats" not in columns:
                logger.info("Adding quality_stats column to training_datasets...")
                session.exec(
                    text("ALTER TABLE training_datasets ADD COLUMN quality_stats TEXT DEFAULT '{}'")
                )
                logger.info("✓ Added quality_stats column")
            else:
                logger.info("quality_stats column already exists")

            session.commit()
            logger.info("Migration completed successfully!")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            session.rollback()
            raise


if __name__ == "__main__":
    migrate_sqlite()
