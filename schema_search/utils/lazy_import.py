from typing import Any
from importlib import import_module


def lazy_import_check(module_name: str, extra_name: str, feature: str) -> Any:
    """
    Lazily import a module and provide helpful error if missing.

    Args:
        module_name: Python module to import (e.g., "sentence_transformers")
        extra_name: pip extra name (e.g., "semantic")
        feature: User-facing feature description (e.g., "semantic search")

    Returns:
        Imported module

    Raises:
        ImportError: With installation instructions if module not found
    """
    try:
        return import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"'{module_name}' is required for {feature}. "
            f"Install with: pip install schema-search[{extra_name}]"
        ) from e
