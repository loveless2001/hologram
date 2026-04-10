#!/usr/bin/env python3
"""Strict temporal query demo on TimeQA.

Builds a year-filtered corpus and answers a query using extractive retrieval.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List

from datasets import load_dataset

from hologram.api import Hologram


YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "it", "that", "this",
    "as", "from", "about", "into", "over", "after", "before", "than", "then",
    "which", "who", "whom", "what", "when", "where", "why", "how",
}


@dataclass
class Row:
    idx: str
    year: int
    context: str


def extract_year(question: str, context: str) -> int:
    q = [int(y) for y in YEAR_RE.findall(question)]
    if q:
        return max(q)
    c = [int(y) for y in YEAR_RE.findall(context)]
    if c:
        return min(c)
    return 1900


def split_sentences(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def chunk_text(text: str, sents_per_chunk: int = 3) -> List[str]:
    s = split_sentences(text)
    if not s:
        return [text[:800]] if text else []
    return [" ".join(s[i : i + sents_per_chunk]) for i in range(0, len(s), sents_per_chunk)]


def keywords(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in toks if t not in STOPWORDS}


def extractive_answer(question: str, contexts: List[str], max_sentences: int = 2) -> str:
    q = keywords(question)
    scored = []
    for ctx in contexts:
        for sent in split_sentences(ctx):
            s = keywords(sent)
            if not s:
                continue
            overlap = len(q & s) / (len(q) or 1)
            density = len(q & s) / len(s)
            scored.append((0.7 * overlap + 0.3 * density, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    seen = set()
    for _, sent in scored:
        k = sent.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(sent)
        if len(out) >= max_sentences:
            break
    return " ".join(out) if out else "Unknown from filtered context."


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict temporal query demo")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--entity-prefix", default="/wiki/Brazilian_Academy_of_Sciences")
    parser.add_argument("--question", required=True)
    parser.add_argument("--year-max", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    ds = load_dataset("hugosousa/TimeQA", split=args.split)
    rows: List[Row] = []
    for r in ds:
        idx = str(r.get("idx", ""))
        if not idx.startswith(args.entity_prefix + "#"):
            continue
        q = str(r.get("question", ""))
        c = str(r.get("context", ""))
        y = extract_year(q, c)
        if y <= args.year_max:
            rows.append(Row(idx=idx, year=y, context=c))

    rows.sort(key=lambda x: (x.year, x.idx))
    hg = Hologram.init(encoder_mode="minilm", use_clip=False, use_gravity=False, auto_ingest_system=False)

    for row in rows:
        gid = f"time:{row.idx}"
        hg.glyphs.create(gid, title=row.idx)
        for i, ch in enumerate(chunk_text(row.context, 3), start=1):
            hg.add_text(
                glyph_id=gid,
                text=ch,
                trace_id=f"{row.idx}:doc:{i}",
                do_extract_concepts=False,
                add_to_field=False,
                skip_nlp=True,
                origin="benchmark",
            )

    hits = hg.search_text(args.question, top_k=args.top_k)
    contexts = [tr.content for tr, _ in hits]
    ans = extractive_answer(args.question, contexts)

    print(f"Temporal mode: hard_filter(year <= {args.year_max})")
    print(f"Rows indexed: {len(rows)}")
    print(f"Question: {args.question}")
    print(f"Answer: {ans}")


if __name__ == "__main__":
    main()

