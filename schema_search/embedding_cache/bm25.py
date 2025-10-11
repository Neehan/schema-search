from typing import List
import re

from rank_bm25 import BM25Okapi

from schema_search.chunkers import Chunk


def light_stem(token: str) -> str:
    """Tiny rule-based stemmer for schema tokens."""
    for suf in ("ing", "ers", "ies", "ied", "ed", "es", "s"):
        if token.endswith(suf) and len(token) > len(suf) + 2:
            if suf == "ies":
                return token[:-3] + "y"
            return token[: -len(suf)]
    return token


def _tokenize(text: str) -> List[str]:
    """Tokenize and normalize database-like text."""
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-z])", r"\1 \2", text)

    tokens = re.findall(r"[a-z0-9]+", text)
    normalized = []
    for t in tokens:
        if t in {"pk", "pkey", "key"}:
            t = "id"
        elif t in {"ts", "time", "timestamp"}:
            t = "timestamp"
        elif t.endswith("id") and len(t) > 2:
            t = "id"
        elif t in {"ix", "index", "idx"}:
            t = "index"
        normalized.append(light_stem(t))
    return normalized


class BM25Cache:
    def __init__(self):
        self.bm25 = None
        self.tokenized_docs = None

    def build(self, chunks: List[Chunk]) -> None:
        if self.bm25 is None:
            self.tokenized_docs = [_tokenize(chunk.content) for chunk in chunks]
            self.bm25 = BM25Okapi(self.tokenized_docs)

    def get_scores(self, query: str):
        if self.bm25 is None:
            raise RuntimeError("BM25 cache not built. Call build() first.")
        return self.bm25.get_scores(_tokenize(query))
