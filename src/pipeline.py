import argparse
import gc
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from .config import (
    TRAIN_S1,
    TRAIN_S2,
    TRAIN_S3,
    TRAIN_GT,
    TEST_S1,
    TEST_S2,
    TEST_S3,
    OUTPUT_DIR,
    MODELS_DIR,
)
from .preprocess import load_and_preprocess
from .blocking import generate_candidates, candidates_to_tsv
from .embeddings import train_word2vec, build_corpus, embed_dataframe
from .matching import (
    build_training_pairs,
    train_with_optuna,
    predict_matches,
    matches_to_tsv,
)

WORD2VEC_PATH = MODELS_DIR / "word2vec_cbow.model"
CLASSIFIER_PATH = MODELS_DIR / "lightgbm_matcher.pkl"


def _save_matcher(classifier):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=MODELS_DIR, suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            pickle.dump(
                {"classifier": classifier, "fallback": classifier is None},
                temporary_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary_path.replace(CLASSIFIER_PATH)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(f"Matcher saved -> {CLASSIFIER_PATH}")


def _load_matcher():
    if not CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(
            f"Saved matcher not found: {CLASSIFIER_PATH}. "
            "Run the training pipeline once before using --test-only."
        )
    with CLASSIFIER_PATH.open("rb") as saved_file:
        saved = pickle.load(saved_file)
    if not isinstance(saved, dict) or "classifier" not in saved or "fallback" not in saved:
        raise ValueError(f"Invalid saved matcher file: {CLASSIFIER_PATH}")
    return saved["classifier"]


def _run_test(model_w2v, classifier, nrows=None):
    print("\n" + "=" * 60)
    print("TEST set")
    print("=" * 60)
    t1 = load_and_preprocess(TEST_S1, nrows=nrows)
    t2 = load_and_preprocess(TEST_S2, nrows=nrows)
    t3 = load_and_preprocess(TEST_S3, nrows=nrows)
    print(f"Test S1={len(t1)}  S2={len(t2)}  S3={len(t3)}")

    test_candidates = generate_candidates(t1, t2, t3)
    candidates_to_tsv(
        test_candidates,
        str(OUTPUT_DIR / "candidate_pairs.tsv"),
        list(t1["entity_id"]),
    )
    emb_t1 = embed_dataframe(t1, model_w2v)
    emb_t2 = embed_dataframe(t2, model_w2v)
    emb_t3 = embed_dataframe(t3, model_w2v)

    if classifier is not None:
        test_matches = predict_matches(
            t1, t2, t3, test_candidates, emb_t1, emb_t2, emb_t3, classifier
        )
    else:
        test_matches = {key: values[:1] for key, values in test_candidates.items()}

    matches_to_tsv(
        test_matches,
        str(OUTPUT_DIR / "output.tsv"),
        list(t1["entity_id"]),
    )
    matches_to_tsv(
        test_matches,
        str(OUTPUT_DIR / "matching_results.tsv"),
        list(t1["entity_id"]),
    )


def run_pipeline(nrows=None, reuse_word2vec=False, test_only=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if test_only:
        if not WORD2VEC_PATH.is_file():
            raise FileNotFoundError(f"Saved Word2Vec model not found: {WORD2VEC_PATH}")
        model_w2v = Word2Vec.load(str(WORD2VEC_PATH))
        classifier = _load_matcher()
        _run_test(model_w2v, classifier, nrows=nrows)
        print("\nDONE. Outputs in:", OUTPUT_DIR)
        return classifier

    print("=" * 60)
    print("1. Loading + Preprocessing training data")
    print("=" * 60)
    s1 = load_and_preprocess(TRAIN_S1, nrows=nrows)
    s2 = load_and_preprocess(TRAIN_S2, nrows=nrows)
    s3 = load_and_preprocess(TRAIN_S3, nrows=nrows)
    gt = pd.read_csv(
        TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, nrows=nrows
    )
    print(f"S1={len(s1)}  S2={len(s2)}  S3={len(s3)}  GT={len(gt)}")

    print("\n" + "=" * 60)
    print("2. Blocking")
    print("=" * 60)
    candidates = generate_candidates(s1, s2, s3)
    avg_candidates = np.mean([len(values) for values in candidates.values()])
    print(f"Avg candidates per S1: {avg_candidates:.1f}")
    candidates_to_tsv(
        candidates,
        str(OUTPUT_DIR / "train_candidate_pairs.tsv"),
        list(s1["entity_id"]),
    )

    print("\n" + "=" * 60)
    print("3. Word2Vec CBOW")
    print("=" * 60)
    if reuse_word2vec:
        if not WORD2VEC_PATH.is_file():
            raise FileNotFoundError(f"Saved Word2Vec model not found: {WORD2VEC_PATH}")
        model_w2v = Word2Vec.load(str(WORD2VEC_PATH))
        print(f"Reusing Word2Vec -> {WORD2VEC_PATH}")
    else:
        corpus = build_corpus([s1, s2, s3])
        print(f"Corpus sentences: {len(corpus)}")
        model_w2v = train_word2vec(corpus, save_path=WORD2VEC_PATH)
        del corpus

    print("\n" + "=" * 60)
    print("4. Embeddings")
    print("=" * 60)
    emb_s1 = embed_dataframe(s1, model_w2v)
    emb_s2 = embed_dataframe(s2, model_w2v)
    emb_s3 = embed_dataframe(s3, model_w2v)

    print("\n" + "=" * 60)
    print("5. Optuna + LightGBM")
    print("=" * 60)
    X, y = build_training_pairs(
        s1, s2, s3, gt, candidates, emb_s1, emb_s2, emb_s3
    )
    if len(y) < 50 or y.sum() < 5:
        print("WARNING: too few positive pairs – using simple top-1 fallback")
        classifier = None
    else:
        classifier = train_with_optuna(X, y)
    _save_matcher(classifier)

    print("\n" + "=" * 60)
    print("6. Train inference")
    print("=" * 60)
    if classifier is not None:
        train_matches = predict_matches(
            s1, s2, s3, candidates, emb_s1, emb_s2, emb_s3, classifier
        )
    else:
        train_matches = {key: values[:1] for key, values in candidates.items()}

    matches_to_tsv(
        train_matches,
        str(OUTPUT_DIR / "train_matching_results.tsv"),
        list(s1["entity_id"]),
    )

    del s1, s2, s3, gt, candidates, emb_s1, emb_s2, emb_s3
    del X, y, train_matches
    gc.collect()

    _run_test(model_w2v, classifier, nrows=nrows)
    print("\nDONE. Outputs in:", OUTPUT_DIR)
    return classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument(
        "--reuse-word2vec",
        action="store_true",
        help="Load models/word2vec_cbow.model instead of retraining Word2Vec.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run test inference from saved Word2Vec and matcher artifacts.",
    )
    args = parser.parse_args()
    run_pipeline(
        nrows=args.nrows,
        reuse_word2vec=args.reuse_word2vec,
        test_only=args.test_only,
    )
