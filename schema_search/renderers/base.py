"""
Base renderer abstract class.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schema_search.types import SearchResult


class BaseRenderer(ABC):
    """Abstract base class for rendering search results."""

    @abstractmethod
    def render(self, search_result: "SearchResult") -> str:
        """
        Render search results to string.

        Args:
            search_result: SearchResult object to render

        Returns:
            Formatted string representation
        """
        raise NotImplementedError
