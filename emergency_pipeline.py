"""
EMERGENCY standalone pipeline - bypasses the slow/heavy preprocess.py and
memory-exploding trigram inverted index entirely.

Run from the project ROOT (same folder as dataset_original/, models/, output/):
    python emergency_pipeline.py

What it does differently (why it's fast + memory-safe):
  - Reads TSVs with plain csv (no pandas), keeps only 3 cleaned strings per
    record (name_clean, addr_clean, country_norm) - NOT full token lists /
    ngram sets per row like preprocess.py did. That per-row bloat across
    11.7M rows is almost certainly what caused the 2-hour preprocessing +
    swap-thrashing.
  - Blocking key = country + first-4-chars-of-cleaned-name (dict of small
    lists), NOT a full character-trigram inverted index (which was building
    ~100M+ set entries and blew memory on Source3).
  - Skips Word2Vec entirely (w2v_cosine feature fixed at 0.0) - saves a lot
    of time; the other 12 features still drive the trained classifier.
  - Batches predict_proba calls (thousands of pairs per call) instead of
    one call per pair.
  - Writes candidate_pairs.tsv row-by-row as it goes (streaming), never
    holds the full candidate set in memory.

Prints progress every 100k rows loaded / 50k S1 entities matched so you can
see it's alive.
"""
import csv
import pickle
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

csv.field_size_limit(2**31 - 1)

# ---------------------------------------------------------------- paths ---
ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "dataset_original" / "test"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "lightgbm_matcher.pkl"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

MAX_CANDIDATES = 15
TOP_K = 4
THRESHOLD = 0.52
BATCH_SIZE = 20000
PREFIX_LEN = 4

# ------------------------------------------------------------ cleaning ---
LEGAL_SUFFIXES = {
    "ltd", "limited", "pvt", "private", "corp", "corporation", "inc", "incorporated",
    "llc", "llp", "plc", "co", "company", "enterprises", "industries", "services",
    "group", "holdings", "international", "intl", "pvt ltd", "private limited",
    "pvt.ltd", "pvt. ltd", "pvt ltd.", "private ltd", "ltd.", "inc.", "corp.",
    "llc.", "llp.", "pc", "p.c.", "dba", "d/b/a", "lp", "l.p.", "pllc",
}
ABBREV_MAP = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "pl": "place",
    "hwy": "highway", "fwy": "freeway", "pkwy": "parkway",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "&": "and", "pvt": "private", "ltd": "limited", "corp": "corporation",
    "inc": "incorporated", "co": "company", "intl": "international",
    "no": "number", "nr": "near", "opp": "opposite", "bldg": "building",
    "sec": "sector", "ph": "phase", "ext": "extension", "extn": "extension",
}
RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_MULTI_SPACE = re.compile(r"\s+")
RE_SPECIAL = re.compile(r"[#@*$%^~`|\\/<>{}\[\]()\"']+")

try:
    from unidecode import unidecode
    HAVE_UNIDECODE = True
except ImportError:
    HAVE_UNIDECODE = False


def normalize_unicode(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    if HAVE_UNIDECODE:
        text = unidecode(text)
    return text.lower().strip()


def clean_text(text):
    text = normalize_unicode(text)
    if not text:
        return ""
    text = RE_SPECIAL.sub(" ", text)
    text = RE_PUNCT.sub(" ", text)
    return RE_MULTI_SPACE.sub(" ", text).strip()


def tokenize_name(name):
    text = clean_text(name)
    if not text:
        return []
    tokens = text.split()
    tokens = [ABBREV_MAP.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return [t for t in tokens if len(t) > 1 and not t.isdigit()]


def tokenize_address(addr):
    text = clean_text(addr)
    if not text:
        return []
    return [ABBREV_MAP.get(t, t) for t in text.split()]


# --------------------------------------------------------------- I/O ------
def load_source(path):
    ids, names, addrs, countries = [], [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        n = 0
        for row in reader:
            if not row:
                continue
            eid = row[idx["entity_id"]]
            name = row[idx["business_name"]] if "business_name" in idx and idx["business_name"] < len(row) else ""
            addr = row[idx["business_address"]] if "business_address" in idx and idx["business_address"] < len(row) else ""
            country = row[idx["country"]] if "country" in idx and idx["country"] < len(row) else ""
            ids.append(eid)
            names.append(" ".join(tokenize_name(name)))
            addrs.append(" ".join(tokenize_address(addr)))
            countries.append(clean_text(country))
            n += 1
            if n % 300000 == 0:
                print(f"  loaded {n} rows from {path.name}", flush=True)
    return ids, names, addrs, countries


def build_blocks(names, countries):
    blocks = defaultdict(list)
    for i, (name, country) in enumerate(zip(names, countries)):
        key = f"{country}|{name[:PREFIX_LEN]}"
        blocks[key].append(i)
    return blocks


def token_set(s):
    return set(s.split()) if s else set()


def sim_score(n1, a1, n2, a2):
    n1s, n2s = token_set(n1), token_set(n2)
    a1s, a2s = token_set(a1), token_set(a2)
    name_ov = len(n1s & n2s) / max(len(n1s | n2s), 1)
    addr_ov = len(a1s & a2s) / max(len(a1s | a2s), 1)
    return 0.65 * name_ov + 0.35 * addr_ov


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("Loading S1...", flush=True)
    s1_ids, s1_name, s1_addr, s1_country = load_source(TEST_S1)
    print("Loading S2...", flush=True)
    s2_ids, s2_name, s2_addr, s2_country = load_source(TEST_S2)
    print("Loading S3...", flush=True)
    s3_ids, s3_name, s3_addr, s3_country = load_source(TEST_S3)
    print(f"S1={len(s1_ids)}  S2={len(s2_ids)}  S3={len(s3_ids)}", flush=True)

    print("Building blocks (memory-light)...", flush=True)
    blocks2 = build_blocks(s2_name, s2_country)
    blocks3 = build_blocks(s3_name, s3_country)

    classifier = None
    if CLASSIFIER_PATH.is_file():
        with CLASSIFIER_PATH.open("rb") as f:
            saved = pickle.load(f)
        classifier = saved.get("classifier") if isinstance(saved, dict) else None
        print(f"Classifier loaded: {classifier is not None}", flush=True)
    else:
        print("WARNING: no saved classifier found, using heuristic fallback.", flush=True)

    from rapidfuzz import fuzz
    FEATURE_NAMES = [
        "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_jaccard",
        "addr_ratio", "addr_partial", "addr_token_sort", "addr_jaccard",
        "w2v_cosine", "country_match", "name_len_diff", "addr_len_diff",
    ]

    def jac(x, y):
        if not x and not y:
            return 1.0
        if not x or not y:
            return 0.0
        return len(x & y) / len(x | y)

    def pair_feats(n1, a1, c1, n2, a2, c2):
        n1s, n2s = token_set(n1), token_set(n2)
        a1s, a2s = token_set(a1), token_set(a2)
        return [
            fuzz.ratio(n1, n2) / 100.0,
            fuzz.partial_ratio(n1, n2) / 100.0,
            fuzz.token_sort_ratio(n1, n2) / 100.0,
            fuzz.token_set_ratio(n1, n2) / 100.0,
            jac(n1s, n2s),
            fuzz.ratio(a1, a2) / 100.0,
            fuzz.partial_ratio(a1, a2) / 100.0,
            fuzz.token_sort_ratio(a1, a2) / 100.0,
            jac(a1s, a2s),
            0.0,  # w2v_cosine skipped
            1.0 if c1 and c2 and c1 == c2 else 0.0,
            abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1),
            abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1),
        ]

    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
    match_path = OUTPUT_DIR / "matching_results.tsv"

    results = {}  # sid -> list[(prob, cid)]  (only matched ones need to be here)
    buf_sid, buf_cid, buf_feat = [], [], []

    def flush():
        if not buf_feat:
            return
        X = np.array(buf_feat, dtype=np.float32)
        probs = classifier.predict_proba(X)[:, 1]
        for sid, cid, p in zip(buf_sid, buf_cid, probs):
            if p >= THRESHOLD:
                results.setdefault(sid, []).append((float(p), cid))
        buf_sid.clear()
        buf_cid.clear()
        buf_feat.clear()

    with candidate_path.open("w", newline="", encoding="utf-8") as cf:
        cw = csv.writer(cf, delimiter="\t", lineterminator="\n")
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])

        n_done = 0
        total = len(s1_ids)
        for i, (sid, name, addr, country) in enumerate(zip(s1_ids, s1_name, s1_addr, s1_country)):
            n_done += 1
            if n_done % 50000 == 0:
                print(f"  Matching: {n_done}/{total}", flush=True)

            key = f"{country}|{name[:PREFIX_LEN]}"
            idxs2 = blocks2.get(key, [])
            idxs3 = blocks3.get(key, [])

            scored = []
            for j in idxs2:
                sc = sim_score(name, addr, s2_name[j], s2_addr[j])
                scored.append((sc, "S2", j))
            for j in idxs3:
                sc = sim_score(name, addr, s3_name[j], s3_addr[j])
                scored.append((sc, "S3", j))
            scored.sort(reverse=True)
            top = scored[:MAX_CANDIDATES]

            cand_ids = [(s2_ids[j] if src == "S2" else s3_ids[j]) for _, src, j in top]
            cw.writerow([sid, ",".join(cand_ids)])

            if classifier is not None:
                for _, src, j in top:
                    if src == "S2":
                        n2, a2, c2 = s2_name[j], s2_addr[j], s2_country[j]
                        cid = s2_ids[j]
                    else:
                        n2, a2, c2 = s3_name[j], s3_addr[j], s3_country[j]
                        cid = s3_ids[j]
                    buf_sid.append(sid)
                    buf_cid.append(cid)
                    buf_feat.append(pair_feats(name, addr, country, n2, a2, c2))
                    if len(buf_feat) >= BATCH_SIZE:
                        flush()
            else:
                if top and top[0][0] >= 0.5:
                    _, src, j = top[0]
                    cid = s2_ids[j] if src == "S2" else s3_ids[j]
                    results[sid] = [(1.0, cid)]

        flush()

    print("Writing matching_results.tsv...", flush=True)
    with match_path.open("w", newline="", encoding="utf-8") as mf:
        mw = csv.writer(mf, delimiter="\t", lineterminator="\n")
        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        for sid in s1_ids:
            scored = results.get(sid, [])
            scored.sort(reverse=True)
            matched = [cid for _, cid in scored[:TOP_K]]
            mw.writerow([sid, ",".join(matched)])

    print("DONE. Outputs written to:", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()