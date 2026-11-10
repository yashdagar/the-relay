from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

from relay.eval import normalize
from relay.library import CHUNK_DIR, ROOT, Chunk, load_chunks

MODEL_CACHE = ROOT / ".cache" / "fastembed"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
RERANK_CANDIDATES = 20
EXCLUDED_SECTIONS = {"References"}

STOPWORDS = set(
    "a an and are as at be by did do does for from how in is it of on or that the this to "
    "use used uses was were what when where which who why with".split()
)


@dataclass(frozen=True)
class RetrievalSetting:
    piece_size: int
    top_k: int
    method: Literal["bm25", "hybrid"]
    rerank: bool = False

    @property
    def name(self) -> str:
        return f"{self.piece_size}w-k{self.top_k}-{self.method}{'-rerank' if self.rerank else ''}"


@dataclass
class Hit:
    chunk: Chunk
    score: float


def _tokens(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t not in STOPWORDS]


@cache
def _embedder() -> TextEmbedding:
    return TextEmbedding(EMBEDDING_MODEL, cache_dir=str(MODEL_CACHE))


@cache
def _reranker() -> TextCrossEncoder:
    return TextCrossEncoder(RERANK_MODEL, cache_dir=str(MODEL_CACHE))


@cache
def _chunk_embeddings(piece_size: int) -> np.ndarray:
    path = CHUNK_DIR / f"embeddings_{piece_size}.npy"
    if path.exists():
        return np.load(path)
    texts = [c.text for c in load_chunks(piece_size)]
    vectors = np.array(list(_embedder().passage_embed(texts)), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(path, vectors)
    return vectors


def _rank(scores: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    return order[allowed[order]]


class Retriever:
    def __init__(self, setting: RetrievalSetting):
        self.setting = setting
        all_chunks = load_chunks(setting.piece_size)
        keep = [i for i, c in enumerate(all_chunks) if c.section not in EXCLUDED_SECTIONS]
        self.chunks = [all_chunks[i] for i in keep]
        self.papers = np.array([c.paper for c in self.chunks])
        self.bm25 = BM25Okapi([_tokens(c.text) for c in self.chunks])
        if setting.method == "hybrid":
            self.embeddings = _chunk_embeddings(setting.piece_size)[keep]
            next(_embedder().query_embed(["warm up"]))
        if setting.rerank:
            list(_reranker().rerank("warm up", ["warm up"]))

    def search(self, query: str, papers: list[str] | None = None) -> list[Hit]:
        allowed = np.isin(self.papers, papers) if papers else np.ones(len(self.chunks), bool)
        bm25_order = _rank(self.bm25.get_scores(_tokens(query)), allowed)
        if self.setting.method == "bm25":
            order, scores = bm25_order, self.bm25.get_scores(_tokens(query))
        else:
            order, scores = self._fuse(query, bm25_order, allowed)
        if self.setting.rerank:
            order, scores = self._rerank(query, order[:RERANK_CANDIDATES])
        top = order[: self.setting.top_k]
        return [Hit(chunk=self.chunks[i], score=float(scores[i])) for i in top]

    def _fuse(self, query: str, bm25_order: np.ndarray, allowed: np.ndarray):
        query_vector = next(_embedder().query_embed([query]))
        dense_order = _rank(self.embeddings @ query_vector, allowed)
        fused = np.zeros(len(self.chunks))
        for order in (bm25_order, dense_order):
            fused[order] += 1 / (RRF_K + np.arange(1, len(order) + 1))
        return _rank(fused, allowed), fused

    def _rerank(self, query: str, candidates: np.ndarray):
        scores = np.full(len(self.chunks), -np.inf)
        texts = [self.chunks[i].text for i in candidates]
        scores[candidates] = list(_reranker().rerank(query, texts))
        return candidates[np.argsort(-scores[candidates], kind="stable")], scores


SETTINGS = {
    s.name: s
    for s in (
        RetrievalSetting(piece_size=120, top_k=8, method="bm25"),
        RetrievalSetting(piece_size=300, top_k=5, method="bm25"),
        RetrievalSetting(piece_size=300, top_k=5, method="hybrid"),
        RetrievalSetting(piece_size=300, top_k=5, method="hybrid", rerank=True),
        RetrievalSetting(piece_size=600, top_k=3, method="hybrid"),
    )
}
DEFAULT_SETTING = SETTINGS["300w-k5-bm25"]
