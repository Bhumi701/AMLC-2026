"""
MEMORY-SAFE + BATCHED drop-in replacement for matching.py.

Two fixes over the original:

1. Same per-row-dict memory bomb as blocking.py (`.to_dict("index")`) -
   replaced with the same compact array-based index (imported from
   blocking.py so there's only one implementation).

2. `predict_matches` used to call `model.predict_proba(...)` ONE PAIR AT A
   TIME. At full test scale (millions of S1-candidate pairs) that is tens
   of millions of tiny python-level calls into LightGBM - easily hours.
   This version batches pairs into chunks (default 20k rows) and calls
   predict_proba once per chunk, which is orders of magnitude faster.

Same function names/signatures as before, so pipeline.py needs no changes.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split

from .features import pair_features, FEATURE_NAMES
from .config import SEED, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, TOP_K_MATCHES, SIMILARITY_THRESHOLD
from .blocking import build_compact_index


def _row_dict(name_arr, addr_arr, country_arr, i):
    return {
        "name_clean": name_arr[i],
        "addr_clean": addr_arr[i],
        "country_norm": country_arr[i],
    }


def build_training_pairs(
    s1_df, s2_df, s3_df, gt, candidates, emb_s1, emb_s2, emb_s3
) -> Tuple[np.ndarray, np.ndarray]:
    id1, name1, addr1, country1 = build_compact_index(s1_df)
    id2, name2, addr2, country2 = build_compact_index(s2_df)
    id3, name3, addr3, country3 = build_compact_index(s3_df)

    true_matches = {}
    for _, row in gt.iterrows():
        sid = row["source1_entity_id"]
        mids = str(row.get("matched_entity_ids") or "").strip()
        true_matches[sid] = set(mids.split(",")) if mids else set()

    X_list, y_list = [], []
    for sid, cands in candidates.items():
        i1 = id1.get(sid)
        if i1 is None:
            continue
        e1 = emb_s1.get(sid)
        if e1 is None:
            continue
        s1r = _row_dict(name1, addr1, country1, i1)
        positives = true_matches.get(sid, set())
        for cid in cands:
            if cid.startswith("S2-"):
                i2 = id2.get(cid)
                e2 = emb_s2.get(cid)
                s2r = _row_dict(name2, addr2, country2, i2) if i2 is not None else None
            else:
                i2 = id3.get(cid)
                e2 = emb_s3.get(cid)
                s2r = _row_dict(name3, addr3, country3, i2) if i2 is not None else None
            if s2r is None or e2 is None:
                continue
            feats = pair_features(s1r, s2r, e1, e2)
            X_list.append([feats[k] for k in FEATURE_NAMES])
            y_list.append(1 if cid in positives else 0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Training pairs: {len(y)} (pos={y.sum()}, neg={len(y)-y.sum()})")
    return X, y


def f05_score(y_true, y_pred) -> float:
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return (1.25 * prec * rec) / (0.25 * prec + rec + 1e-9)


def optuna_objective(trial, X, y):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": SEED,
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 80, 250),
    }
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y if y.sum() > 10 else None)
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    return f05_score(y_va, pred)


def train_with_optuna(X, y) -> lgb.LGBMClassifier:
    print(f"Running Optuna ({OPTUNA_N_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: optuna_objective(t, X, y), n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)
    print("Best F0.5:", study.best_value)
    print("Best params:", study.best_params)

    best = study.best_params
    best.update({"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": SEED})
    model = lgb.LGBMClassifier(**best)
    model.fit(X, y)
    return model


def predict_matches(
    s1_df, s2_df, s3_df, candidates, emb_s1, emb_s2, emb_s3, model,
    threshold=SIMILARITY_THRESHOLD, top_k=TOP_K_MATCHES, batch_size=20000,
):
    id1, name1, addr1, country1 = build_compact_index(s1_df)
    id2, name2, addr2, country2 = build_compact_index(s2_df)
    id3, name3, addr3, country3 = build_compact_index(s3_df)

    scored_pairs = defaultdict(list)  # sid -> list[(prob, cid)]
    results = {}

    buf_sid, buf_cid, buf_feat = [], [], []

    def flush():
        if not buf_feat:
            return
        X = np.array(buf_feat, dtype=np.float32)
        probs = model.predict_proba(X)[:, 1]
        for sid, cid, p in zip(buf_sid, buf_cid, probs):
            if p >= threshold:
                scored_pairs[sid].append((float(p), cid))
        buf_sid.clear()
        buf_cid.clear()
        buf_feat.clear()

    n_done = 0
    total = len(candidates)
    for sid, cands in candidates.items():
        n_done += 1
        if n_done % 50000 == 0:
            print(f"  Predicting: {n_done}/{total}")

        i1 = id1.get(sid)
        if i1 is None:
            results[sid] = []
            continue
        e1 = emb_s1.get(sid)
        if e1 is None:
            results[sid] = []
            continue
        s1r = _row_dict(name1, addr1, country1, i1)

        for cid in cands:
            if cid.startswith("S2-"):
                i2 = id2.get(cid)
                e2 = emb_s2.get(cid)
                s2r = _row_dict(name2, addr2, country2, i2) if i2 is not None else None
            else:
                i2 = id3.get(cid)
                e2 = emb_s3.get(cid)
                s2r = _row_dict(name3, addr3, country3, i2) if i2 is not None else None
            if s2r is None or e2 is None:
                continue
            feats = pair_features(s1r, s2r, e1, e2)
            buf_sid.append(sid)
            buf_cid.append(cid)
            buf_feat.append([feats[k] for k in FEATURE_NAMES])
            if len(buf_feat) >= batch_size:
                flush()

    flush()  # last partial batch

    for sid in candidates.keys():
        if sid in results:
            continue
        scored = scored_pairs.get(sid, [])
        scored.sort(reverse=True)
        results[sid] = [cid for _, cid in scored[:top_k]]
    return results


def matches_to_tsv(matches: Dict[str, List[str]], out_path: str, s1_ids: List[str]):
    rows = []
    for sid in s1_ids:
        mids = matches.get(sid, [])
        rows.append({
            "source1_entity_id": sid,
            "matched_entity_ids": ",".join(mids) if mids else ""
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")