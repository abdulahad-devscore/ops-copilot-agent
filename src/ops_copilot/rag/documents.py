"""Load markdown documents with YAML front matter and split them into chunks.

Chunking is heading-aware: a chunk never spans two sections, every chunk
carries its document title and section path (so "Credit hold" chunks still
say they come from the Customer Credit Policy), and tables are kept whole
where they fit. Retrieval quality is decided here more than in the prompt.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROLES = ("staff", "store_manager", "finance", "management")


@dataclass
class Document:
    id: str
    title: str
    doc_type: str
    access: list[str]
    effective: str
    path: str
    body: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    doc_type: str
    section: str
    text: str
    access: list[str]
    effective: str
    content_hash: str = field(default="")

    def embed_text(self) -> str:
        """What gets embedded: context first, so short chunks still carry meaning."""
        return f"{self.title}\n{self.section}\n\n{self.text}"


def load_corpus(root: str | Path) -> list[Document]:
    docs = []
    for path in sorted(Path(root).rglob("*.md")):
        raw = path.read_text()
        if not raw.startswith("---"):
            continue
        _, meta_text, body = raw.split("---", 2)
        meta = yaml.safe_load(meta_text)
        access = meta.get("access") or list(ROLES)
        unknown = set(access) - set(ROLES)
        if unknown:
            raise ValueError(f"{path}: unknown roles {unknown}")
        docs.append(
            Document(
                id=str(meta["id"]),
                title=meta["title"],
                doc_type=meta.get("type", "doc"),
                access=list(access),
                effective=str(meta.get("effective", "")),
                path=str(path.relative_to(root)),
                body=body.strip(),
            )
        )
    ids = [d.id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate document ids: {dupes}")
    return docs


_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")


def _sections(body: str) -> list[tuple[str, str]]:
    """Split on headings; returns (section path, text) pairs."""
    path: list[str] = []
    current: list[str] = []
    out: list[tuple[str, str]] = []

    def flush() -> None:
        text = "\n".join(current).strip()
        if text:
            out.append((" > ".join(path), text))
        current.clear()

    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            current.append(line)
    flush()
    return out


def _split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split a section into paragraph-aligned windows; tables stay intact when they fit."""
    if len(text) <= max_chars:
        return [text]
    blocks: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        blocks.extend(_split_table(block, max_chars) if block.lstrip().startswith("|") else [block])
    pieces: list[str] = []
    buf = ""
    for block in blocks:
        if buf and len(buf) + len(block) + 2 > max_chars:
            pieces.append(buf)
            tail = buf[-overlap:]
            # Overlap helps prose; for tables it would start a chunk mid-row, so skip it.
            table_edge = block.lstrip().startswith("|") or buf.rstrip().endswith("|")
            buf = "" if table_edge else (tail[tail.find(" ") + 1 :] if " " in tail else "")
        buf = f"{buf}\n\n{block}".strip() if buf else block
        while len(buf) > max_chars * 1.5:  # one huge paragraph: hard wrap
            pieces.append(buf[:max_chars])
            buf = buf[max_chars - overlap :]
    if buf:
        pieces.append(buf)
    return pieces


def _split_table(block: str, max_chars: int) -> list[str]:
    """Split a markdown table by rows, repeating the header so every piece is a readable table."""
    if len(block) <= max_chars:
        return [block]
    lines = block.strip().splitlines()
    header, rows = lines[:2], lines[2:]
    pieces, current = [], list(header)
    for row in rows:
        if len("\n".join(current)) + len(row) > max_chars and len(current) > 2:
            pieces.append("\n".join(current))
            current = list(header)
        current.append(row)
    pieces.append("\n".join(current))
    return pieces


def chunk_document(doc: Document, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    chunks = []
    for section, text in _sections(doc.body) or [(doc.title, doc.body)]:
        for piece in _split(text, max_chars, overlap):
            n = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{doc.id}#{n}",
                    doc_id=doc.id,
                    title=doc.title,
                    doc_type=doc.doc_type,
                    section=section or doc.title,
                    text=piece,
                    access=doc.access,
                    effective=doc.effective,
                    content_hash=hashlib.sha256(f"{doc.title}|{section}|{piece}".encode()).hexdigest()[:16],
                )
            )
    return chunks
