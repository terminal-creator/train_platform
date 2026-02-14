"""
Config templates for all supported training algorithms.

Each template provides default configurations that can be customized.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent


def get_template(algorithm: str) -> Dict[str, Any]:
    """Load a config template for the given algorithm."""
    template_file = TEMPLATES_DIR / f"{algorithm}.json"
    if not template_file.exists():
        raise FileNotFoundError(f"No template found for algorithm: {algorithm}")
    with open(template_file, "r") as f:
        return json.load(f)


def list_templates() -> list:
    """List all available config templates."""
    templates = []
    for f in sorted(TEMPLATES_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            templates.append({
                "algorithm": f.stem,
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "category": data.get("category", ""),
            })
        except Exception as e:
            logger.warning(f"Failed to load template {f}: {e}")
    return templates


def validate_config(algorithm: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a user config against the template, return merged config with defaults."""
    template = get_template(algorithm)
    defaults = template.get("defaults", {})

    merged = {**defaults, **config}

    # Validate required fields
    required = template.get("required_fields", [])
    missing = [f for f in required if f not in config or config[f] is None]

    return {
        "valid": len(missing) == 0,
        "missing_fields": missing,
        "config": merged,
        "warnings": [],
    }
