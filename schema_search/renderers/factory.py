"""
Factory for creating renderers.
"""

from schema_search.renderers.base import BaseRenderer
from schema_search.renderers.json import JsonRenderer
from schema_search.renderers.markdown import MarkdownRenderer


def create_renderer(output_format: str) -> BaseRenderer:
    """
    Factory function to create a renderer based on output format.

    Args:
        output_format: "json" or "markdown"

    Returns:
        Renderer instance

    Raises:
        ValueError: If output format is not supported
    """
    if output_format == "json":
        return JsonRenderer()
    elif output_format == "markdown":
        return MarkdownRenderer()
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
