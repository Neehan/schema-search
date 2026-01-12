"""
JSON renderer for search results.
"""

import json
from typing import TYPE_CHECKING

from schema_search.renderers.base import BaseRenderer

if TYPE_CHECKING:
    from schema_search.types import SearchResult


class JsonRenderer(BaseRenderer):
    """Renders search results as JSON string."""

    def render(self, search_result: "SearchResult") -> str:
        """Render search results as JSON string."""
        return json.dumps(search_result.to_dict(), indent=2)
