from typing import Dict, Any
import numpy as np
from rapidfuzz import fuzz
from .embeddings import cosine_sim

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def pair_features(s1: Dict[str, Any], s2: Dict[str, Any], emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
    n1 = s1.get("name_clean") or ""
    n2 = s2.get("name_clean") or ""
    a1 = s1.get("addr_clean") or ""
    a2 = s2.get("addr_clean") or ""
    c1 = s1.get("country_norm") or ""
    c2 = s2.get("country_norm") or ""

    name_tok1 = set(n1.split())
    name_tok2 = set(n2.split())
    addr_tok1 = set(a1.split())
    addr_tok2 = set(a2.split())

    return {
        "name_ratio": fuzz.ratio(n1, n2) / 100.0,
        "name_partial": fuzz.partial_ratio(n1, n2) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_token_set": fuzz.token_set_ratio(n1, n2) / 100.0,
        "name_jaccard": jaccard(name_tok1, name_tok2),
        "addr_ratio": fuzz.ratio(a1, a2) / 100.0,
        "addr_partial": fuzz.partial_ratio(a1, a2) / 100.0,
        "addr_token_sort": fuzz.token_sort_ratio(a1, a2) / 100.0,
        "addr_jaccard": jaccard(addr_tok1, addr_tok2),
        "w2v_cosine": cosine_sim(emb1, emb2),
        "country_match": 1.0 if c1 and c2 and c1 == c2 else 0.0,
        "name_len_diff": abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1),
        "addr_len_diff": abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1),
    }

FEATURE_NAMES = [
    "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_jaccard",
    "addr_ratio", "addr_partial", "addr_token_sort", "addr_jaccard",
    "w2v_cosine", "country_match", "name_len_diff", "addr_len_diff",
]