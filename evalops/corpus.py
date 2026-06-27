"""Load a markdown corpus into retrievable chunks.

Each `.md` file under the corpus directory is one document; its filename (without
extension) is the stable `doc_id` used as the retrieval relevance label. Documents
are split into chunks on blank lines so a chunk is roughly a paragraph — small
enough to be a precise citation unit.
"""

from __future__ import annotations

from pathlib import Path


def load_chunks(corpus_dir: str | Path) -> list[dict]:
    corpus_dir = Path(corpus_dir)
    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md documents found in {corpus_dir}")

    chunks: list[dict] = []
    for path in files:
        doc_id = path.stem
        paragraphs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n")]
        for i, para in enumerate(p for p in paragraphs if p):
            chunks.append({"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id, "text": para})
    return chunks
