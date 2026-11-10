import json
import re
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "data" / "papers"
CHUNK_DIR = ROOT / "data" / "chunks"

PAPERS = {
    "1706.03762v7": "Attention Is All You Need",
    "2410.00037v1": "Moshi: a speech-text foundation model for real-time dialogue",
    "2507.07955v2": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
}

SECTION_NUMBER = re.compile(r"^([A-Z]|\d+)(\.\d+)*\.?$")


@dataclass
class Chunk:
    id: str
    paper: str
    section: str
    page: int
    text: str

    def cite(self) -> str:
        return f"[{self.paper} §{self.section} p.{self.page}]"


def _normalise(s: str) -> str:
    return re.sub(r"\W+", " ", s).strip().lower()


def _section_lines(paper: str) -> list[tuple[str, int, str]]:
    doc = pymupdf.open(PAPER_DIR / f"{paper}.pdf")
    toc = [(title, page) for _, title, page in doc.get_toc()]
    pending = list(toc)
    section = "Front matter"
    out = []
    for page_no, page in enumerate(doc, start=1):
        lines = [raw.strip() for raw in page.get_text().splitlines()]
        switched_on_page = False
        for i, line in enumerate(lines):
            if not line:
                continue
            if line == "References" and len(line) < 20:
                section = "References"
            elif pending and pending[0][1] == page_no:
                title, _ = pending[0]
                prev = lines[i - 1] if i else ""
                inline = re.match(r"^([A-Z]|\d+)(\.\d+)*\.?\s+(.*)$", line)
                if _normalise(line) == _normalise(title) and SECTION_NUMBER.match(prev):
                    section, switched_on_page = title, True
                    pending.pop(0)
                elif inline and _normalise(inline.group(3)) == _normalise(title):
                    section, switched_on_page = title, True
                    pending.pop(0)
            out.append((section, page_no, line))
        while pending and pending[0][1] <= page_no and not switched_on_page:
            section = pending.pop(0)[0]
    return out


def build_chunks(size: int, overlap: float = 0.15) -> list[Chunk]:
    chunks = []
    step = max(1, int(size * (1 - overlap)))
    for paper in PAPERS:
        runs: list[tuple[str, list[tuple[int, str]]]] = []
        for section, page, line in _section_lines(paper):
            if not runs or runs[-1][0] != section:
                runs.append((section, []))
            runs[-1][1].extend((page, w) for w in line.split())
        for section, words in runs:
            for start in range(0, max(1, len(words) - size + step), step):
                piece = words[start : start + size]
                if len(piece) < 20 and start:
                    continue
                chunks.append(
                    Chunk(
                        id=f"{paper}:{size}:{len(chunks)}",
                        paper=paper,
                        section=section,
                        page=piece[0][0],
                        text=" ".join(w for _, w in piece),
                    )
                )
    return chunks


@cache
def load_chunks(size: int) -> list[Chunk]:
    path = CHUNK_DIR / f"chunks_{size}.json"
    if path.exists():
        return [Chunk(**c) for c in json.loads(path.read_text())]
    chunks = build_chunks(size)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in chunks]))
    return chunks
