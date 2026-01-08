"""
Database Migration Script for Phase 3

Adds Phase 3 features:
1. Create Pipeline table for multi-stage workflows
2. Create PipelineStage table for pipeline stages

Usage:
    python -m training_platform.core.migrate_phase3
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlmodel import Session
from training_platform.core.database import engine, SQLModel, Pipeline, PipelineStage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """Run Phase 3 migration"""
    logger.info("=" * 80)
    logger.info("Phase 3 Migration: Task System Upgrade")
    logger.info("=" * 80)

    try:
        # Create Pipeline and PipelineStage tables
        logger.info("Creating Pipeline tables...")
        SQLModel.metadata.create_all(
            engine,
            tables=[Pipeline.__table__, PipelineStage.__table__]
        )
        logger.info("✓ Pipeline tables created")

        logger.info("=" * 80)
        logger.info("✓ Phase 3 migration completed successfully!")
        logger.info("=" * 80)
        logger.info("\nNew features:")
        logger.info("  - Pipeline table: Multi-stage workflow orchestration")
        logger.info("  - PipelineStage table: Individual pipeline stages")
        logger.info("  - Celery task integration")
        logger.info("  - Task priority and retry mechanism")

    except Exception as e:
        logger.error("=" * 80)
        logger.error("✗ Migration failed!")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        sys.exit(1)


def main():
    """Main migration entry point"""
    db_url = os.getenv("DATABASE_URL", "sqlite:///./training_platform.db")
    logger.info(f"Database URL: {db_url}")

    migrate()


if __name__ == "__main__":
    main()
