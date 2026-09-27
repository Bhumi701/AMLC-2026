"""
DROP-IN REPLACEMENT for embeddings.py - same function names/signatures,
just fast (no .iterrows()). Rename this file to embeddings.py.
"""
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from .config import (
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT,
    W2V_EPOCHS, W2V_SG, W2V_WORKERS, MODELS_DIR, SEED
)


def build_corpus(dfs: List[pd.DataFrame]) -> List[List[str]]:
    corpus = []
    for df in dfs:
        for text in df["combined_text"].fillna("").values:
            if text:
                corpus.append(text.split())
    return corpus


def train_word2vec(corpus: List[List[str]], save_path: Path = None) -> Word2Vec:
    model = Word2Vec(
        sentences=corpus,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=W2V_SG,               # 0 = CBOW
        workers=W2V_WORKERS,
        epochs=W2V_EPOCHS,
        seed=SEED,
    )
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"Word2Vec saved -> {save_path}")
    return model


def sentence_vector(tokens: List[str], model: Word2Vec) -> np.ndarray:
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def embed_dataframe(df: pd.DataFrame, model: Word2Vec) -> Dict[str, np.ndarray]:
    emb = {}
    for eid, text in zip(df["entity_id"].values, df["combined_text"].fillna("").values):
        tokens = text.split()
        emb[eid] = sentence_vector(tokens, model)
    return emb
