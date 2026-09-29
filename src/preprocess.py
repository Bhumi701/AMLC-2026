"""Memory-conscious preprocessing that preserves the existing output schema."""
import re
import unicodedata
from typing import List, Set, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

PREPROCESS_CHUNK_SIZE = 50_000
OUTPUT_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "name_clean",
    "addr_clean",
    "name_tokens",
    "addr_tokens",
    "country_norm",
    "name_ngrams",
    "combined_text",
)

LEGAL_SUFFIXES = {
    "ltd", "limited", "pvt", "private", "corp", "corporation", "inc", "incorporated",
    "llc", "llp", "plc", "co", "company", "enterprises", "industries", "services",
    "group", "holdings", "international", "intl", "pvt ltd", "private limited",
    "pvt.ltd", "pvt. ltd", "pvt ltd.", "private ltd", "ltd.", "inc.", "corp.",
    "llc.", "llp.", "pc", "p.c.", "dba", "d/b/a", "lp", "l.p.", "pllc"
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
RE_SPECIAL = re.compile(r"[#@*$%^~`|\\/<>{}[\]()\"']+")

try:
    from unidecode import unidecode
    HAVE_UNIDECODE = True
except ImportError:
    HAVE_UNIDECODE = False
    print("WARNING: unidecode not installed (pip install Unidecode) - "
          "non-Latin script names (Devanagari/Bengali/Tamil/...) will NOT "
          "be transliterated, hurting cross-script matches.")


def normalize_unicode(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    if HAVE_UNIDECODE:
        text = unidecode(text)  # romanize any non-Latin script
    return text.lower().strip()


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    if not text:
        return ""
    text = RE_SPECIAL.sub(" ", text)
    text = RE_PUNCT.sub(" ", text)
    return RE_MULTI_SPACE.sub(" ", text).strip()


def expand_abbreviations(tokens: List[str]) -> List[str]:
    return [ABBREV_MAP.get(t, t) for t in tokens]


def remove_legal_suffixes(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in LEGAL_SUFFIXES]


def tokenize_name(name: str) -> List[str]:
    text = clean_text(name)
    if not text:
        return []
    tokens = text.split()
    tokens = expand_abbreviations(tokens)
    tokens = remove_legal_suffixes(tokens)
    return [t for t in tokens if len(t) > 1 and not t.isdigit()]


def tokenize_address(addr: str) -> List[str]:
    text = clean_text(addr)
    if not text:
        return []
    tokens = expand_abbreviations(text.split())
    return [t for t in tokens if t]


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    text = clean_text(text).replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a dataframe without retaining a second list of row dicts."""
    columns = {name: [] for name in OUTPUT_COLUMNS}
    _append_preprocessed_rows(df, columns, 0)
    return _dataframe_from_columns(columns)


def _append_preprocessed_rows(df: pd.DataFrame, columns: Dict[str, list], processed: int) -> int:
    entity_ids = df["entity_id"].values
    names = df["business_name"].fillna("").values if "business_name" in df else [""] * len(df)
    addrs = df["business_address"].fillna("").values if "business_address" in df else [""] * len(df)
    countries = df["country"].fillna("").values if "country" in df else [""] * len(df)

    for eid, name, addr, country in zip(entity_ids, names, addrs, countries):
        processed += 1
        if processed % 200000 == 0:
            print(f"  Preprocessing: {processed} rows")
        name_tokens = tokenize_name(name)
        addr_tokens = tokenize_address(addr)
        country_norm = clean_text(country)
        name_clean = " ".join(name_tokens)
        addr_clean = " ".join(addr_tokens)
        columns["entity_id"].append(eid)
        columns["business_name"].append(name)
        columns["business_address"].append(addr)
        columns["country"].append(country)
        columns["name_clean"].append(name_clean)
        columns["addr_clean"].append(addr_clean)
        columns["name_tokens"].append(name_tokens)
        columns["addr_tokens"].append(addr_tokens)
        columns["country_norm"].append(country_norm)
        columns["name_ngrams"].append(get_char_ngrams(name, 3))
        columns["combined_text"].append(f"{name_clean} {addr_clean} {country_norm}".strip())
    return processed


def _dataframe_from_columns(columns: Dict[str, list]) -> pd.DataFrame:
    arrays = {}
    for name in OUTPUT_COLUMNS:
        values = columns[name]
        arrays[name] = np.fromiter(values, dtype=object, count=len(values))
        del columns[name]
    return pd.DataFrame(arrays, columns=OUTPUT_COLUMNS, copy=False)


def preprocess_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Kept for backward compatibility (matching.py / other callers might
    still import this) - not used by the fast preprocess_dataframe above."""
    name = row.get("business_name", "") or ""
    addr = row.get("business_address", "") or ""
    country = row.get("country", "") or ""
    name_tokens = tokenize_name(name)
    addr_tokens = tokenize_address(addr)
    country_norm = clean_text(country)
    name_clean = " ".join(name_tokens)
    addr_clean = " ".join(addr_tokens)
    return {
        "entity_id": row["entity_id"],
        "business_name": name,
        "business_address": addr,
        "country": country,
        "name_clean": name_clean,
        "addr_clean": addr_clean,
        "name_tokens": name_tokens,
        "addr_tokens": addr_tokens,
        "country_norm": country_norm,
        "name_ngrams": get_char_ngrams(name, 3),
        "combined_text": f"{name_clean} {addr_clean} {country_norm}".strip(),
    }


def load_and_preprocess(path: Path, nrows: int = None) -> pd.DataFrame:
    columns = {name: [] for name in OUTPUT_COLUMNS}
    processed = 0
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
        chunksize=PREPROCESS_CHUNK_SIZE,
    )
    required_columns = ("entity_id", "business_name", "business_address", "country")
    found_columns = False
    for chunk in reader:
        found_columns = True
        for col in required_columns:
            if col not in chunk.columns:
                raise ValueError(f"Missing column {col} in {path}")
        processed = _append_preprocessed_rows(chunk, columns, processed)

    if not found_columns:
        header = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=0)
        for col in required_columns:
            if col not in header.columns:
                raise ValueError(f"Missing column {col} in {path}")

    return _dataframe_from_columns(columns)
