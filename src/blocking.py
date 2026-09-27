"""
MEMORY-SAFE drop-in replacement for blocking.py.

Root cause of the S3 crash: `s2_df.set_index("entity_id").to_dict("index")`
builds ONE PYTHON DICT PER ROW (every column, incl. token lists / ngram
sets) for the entire dataframe. For millions of Source2/Source3 rows that
easily eats 10+ GB even though downstream code only ever reads 3 string
fields (name_clean, addr_clean, country_norm).

This version keeps those 3 columns as plain numpy arrays + a slim
id -> row-index dict (`build_compact_index`), and only builds a tiny
transient dict for the handful of candidates actually being scored per
S1 entity. Same function names/signatures as before.
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .config import MAX_CANDIDATES_PER_ENTITY, BLOCK_NGRAM, COUNTRY_MUST_MATCH
from .preprocess import get_char_ngrams


def build_inverted_index(df: pd.DataFrame, field: str = "name_clean", n: int = BLOCK_NGRAM) -> Dict[str, Set[str]]:
    index = defaultdict(set)
    for eid, text in zip(df["entity_id"].values, df[field].fillna("").values):
        for ng in get_char_ngrams(text, n):
            index[ng].add(eid)
    return index


def build_compact_index(df: pd.DataFrame) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """id -> row index, plus the 3 columns actually needed downstream, as
    plain arrays. NEVER materializes a python dict per row."""
    ids = df["entity_id"].values
    id_to_idx = {eid: i for i, eid in enumerate(ids)}
    name_arr = df["name_clean"].fillna("").values
    addr_arr = df["addr_clean"].fillna("").values
    country_arr = df["country_norm"].fillna("").values
    return id_to_idx, name_arr, addr_arr, country_arr


def candidate_score(s1_row: dict, s2_row: dict) -> float:
    n1 = set((s1_row.get("name_clean") or "").split())
    n2 = set((s2_row.get("name_clean") or "").split())
    a1 = set((s1_row.get("addr_clean") or "").split())
    a2 = set((s2_row.get("addr_clean") or "").split())
    name_overlap = len(n1 & n2) / max(len(n1 | n2), 1)
    addr_overlap = len(a1 & a2) / max(len(a1 | a2), 1)
    return 0.65 * name_overlap + 0.35 * addr_overlap


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_cand: int = MAX_CANDIDATES_PER_ENTITY,
) -> Dict[str, List[str]]:
    print("Building inverted indexes...")
    idx2 = build_inverted_index(s2_df, "name_clean")
    idx3 = build_inverted_index(s3_df, "name_clean")

    print("Building compact (memory-light) row indexes...")
    id2_to_idx, name2, addr2, country2 = build_compact_index(s2_df)
    id3_to_idx, name3, addr3, country3 = build_compact_index(s3_df)

    candidates = {}
    cols = ["entity_id", "name_clean", "country_norm"]
    n_done = 0
    total = len(s1_df)
    for eid, name, country in zip(*(s1_df[c].fillna("").values for c in cols)):
        n_done += 1
        if n_done % 20000 == 0:
            print(f"  Blocking: {n_done}/{total}")

        ngrams = get_char_ngrams(name, BLOCK_NGRAM)
        pot2: Set[str] = set()
        pot3: Set[str] = set()
        for ng in ngrams:
            pot2 |= idx2.get(ng, set())
            pot3 |= idx3.get(ng, set())

        row = {"name_clean": name, "country_norm": country}
        scored = []
        for cid in pot2:
            i = id2_to_idx.get(cid)
            if i is None:
                continue
            c_country = country2[i]
            if COUNTRY_MUST_MATCH and country and c_country and c_country != country:
                continue
            crow = {"name_clean": name2[i], "addr_clean": addr2[i]}
            scored.append((candidate_score(row, crow), cid))
        for cid in pot3:
            i = id3_to_idx.get(cid)
            if i is None:
                continue
            c_country = country3[i]
            if COUNTRY_MUST_MATCH and country and c_country and c_country != country:
                continue
            crow = {"name_clean": name3[i], "addr_clean": addr3[i]}
            scored.append((candidate_score(row, crow), cid))

        scored.sort(reverse=True)
        candidates[eid] = [cid for _, cid in scored[:max_cand]]

    return candidates


def candidates_to_tsv(candidates: Dict[str, List[str]], out_path: str, s1_ids: List[str]):
    rows = []
    for sid in s1_ids:
        cands = candidates.get(sid, [])
        rows.append({
            "source1_entity_id": sid,
            "candidate_entity_ids": ",".join(cands) if cands else ""
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")