"""
Database Migration Script for Phase 2

Adds Phase 2 features:
1. Add recipe_id and dataset_version_hash to TrainingJob
2. Create DatasetVersion table for data lineage tracking

Usage:
    python -m training_platform.core.migrate_phase2
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlmodel import Session, text
from training_platform.core.database import engine, SQLModel, DatasetVersion
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_sqlite():
    """Migration for SQLite database"""
    logger.info("Starting SQLite migration for Phase 2...")

    with Session(engine) as session:
        try:
            # Check if columns already exist
            result = session.exec(text("PRAGMA table_info(training_jobs)"))
            columns = [row[1] for row in result.fetchall()]

            # Add recipe_id column if it doesn't exist
            if "recipe_id" not in columns:
                logger.info("Adding recipe_id column to training_jobs...")
                session.exec(
                    text("ALTER TABLE training_jobs ADD COLUMN recipe_id VARCHAR")
                )
                # Create index
                session.exec(
                    text("CREATE INDEX ix_training_jobs_recipe_id ON training_jobs(recipe_id)")
                )
                logger.info("✓ Added recipe_id column")
            else:
                logger.info("recipe_id column already exists")

            # Add dataset_version_hash column if it doesn't exist
            if "dataset_version_hash" not in columns:
                logger.info("Adding dataset_version_hash column to training_jobs...")
                session.exec(
                    text("ALTER TABLE training_jobs ADD COLUMN dataset_version_hash VARCHAR")
                )
                # Create index
                session.exec(
                    text("CREATE INDEX ix_training_jobs_dataset_version_hash ON training_jobs(dataset_version_hash)")
                )
                logger.info("✓ Added dataset_version_hash column")
            else:
                logger.info("dataset_version_hash column already exists")

            session.commit()
            logger.info("✓ TrainingJob table migration completed")

        except Exception as e:
            logger.error(f"Error migrating training_jobs table: {e}")
            session.rollback()
            raise

    # Create DatasetVersion table
    try:
        logger.info("Creating DatasetVersion table...")
        SQLModel.metadata.create_all(engine, tables=[DatasetVersion.__table__])
        logger.info("✓ DatasetVersion table created")
    except Exception as e:
        logger.error(f"Error creating DatasetVersion table: {e}")
        raise


def migrate_postgresql():
    """Migration for PostgreSQL database"""
    logger.info("Starting PostgreSQL migration for Phase 2...")

    with Session(engine) as session:
        try:
            # Add recipe_id column if it doesn't exist
            logger.info("Adding recipe_id column to training_jobs...")
            session.exec(
                text("""
                    ALTER TABLE training_jobs
                    ADD COLUMN IF NOT EXISTS recipe_id VARCHAR
                """)
            )
            # Create index
            session.exec(
                text("""
                    CREATE INDEX IF NOT EXISTS ix_training_jobs_recipe_id
                    ON training_jobs(recipe_id)
                """)
            )
            logger.info("✓ Added recipe_id column")

            # Add dataset_version_hash column if it doesn't exist
            logger.info("Adding dataset_version_hash column to training_jobs...")
            session.exec(
                text("""
                    ALTER TABLE training_jobs
                    ADD COLUMN IF NOT EXISTS dataset_version_hash VARCHAR
                """)
            )
            # Create index
            session.exec(
                text("""
                    CREATE INDEX IF NOT EXISTS ix_training_jobs_dataset_version_hash
                    ON training_jobs(dataset_version_hash)
                """)
            )
            logger.info("✓ Added dataset_version_hash column")

            session.commit()
            logger.info("✓ TrainingJob table migration completed")

        except Exception as e:
            logger.error(f"Error migrating training_jobs table: {e}")
            session.rollback()
            raise

    # Create DatasetVersion table
    try:
        logger.info("Creating DatasetVersion table...")
        SQLModel.metadata.create_all(engine, tables=[DatasetVersion.__table__])
        logger.info("✓ DatasetVersion table created")
    except Exception as e:
        logger.error(f"Error creating DatasetVersion table: {e}")
        raise


def main():
    """Run migration based on database type"""
    db_url = os.getenv("DATABASE_URL", "sqlite:///./training_platform.db")

    logger.info(f"Database URL: {db_url}")
    logger.info("=" * 80)
    logger.info("Phase 2 Migration: Recipe System + Data Versioning")
    logger.info("=" * 80)

    try:
        if "sqlite" in db_url:
            migrate_sqlite()
        elif "postgresql" in db_url:
            migrate_postgresql()
        else:
            logger.error(f"Unsupported database type: {db_url}")
            sys.exit(1)

        logger.info("=" * 80)
        logger.info("✓ Phase 2 migration completed successfully!")
        logger.info("=" * 80)
        logger.info("\nNew features:")
        logger.info("  - TrainingJob.recipe_id: Link jobs to training recipes")
        logger.info("  - TrainingJob.dataset_version_hash: Track dataset versions")
        logger.info("  - DatasetVersion table: Store dataset snapshots for lineage")

    except Exception as e:
        logger.error("=" * 80)
        logger.error("✗ Migration failed!")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
