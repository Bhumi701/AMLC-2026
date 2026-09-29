# Amazon ML Challenge 2026 — Business Entity Resolution

## Problem
Given noisy business records from 3 independent sources (Source1 = deduplicated
reference, Source2/Source3 = raw), determine which Source2/Source3 records
refer to the same real-world business as each Source1 entity. Evaluated with
macro-averaged F0.5 (precision-weighted).

## Final pipeline — `miti.py`
After repeated crashes with the original modular pipeline (`src/pipeline.py` +
`src/preprocess.py` + `src/blocking.py`) at full test scale (S1=1,732,544,
S2=4,887,273, S3=5,082,316 rows), the final working solution is a single
standalone, memory-light script: **`miti.py`** (originally
`emergency_pipeline.py`).

### Why the original pipeline failed at scale
- `preprocess.py` stored a full token list + character-ngram **set** per row,
  for every one of the ~11.7M records, across all 3 sources. At this scale
  that pushed memory usage high enough to start OS-level swapping, turning a
  step that should take minutes into ~2 hours.
- `blocking.py` built a **full character-trigram inverted index**
  (`ngram -> set(entity_id)`) over Source2/Source3. With millions of records
  this produced 100M+ set entries and raised a `MemoryError` while indexing
  Source3.
- `matching.py` called `model.predict_proba()` **one pair at a time**, which
  at full scale would have meant tens of millions of individual calls —
  far too slow to finish within the hackathon window.

### What `miti.py` does differently
1. **Streaming, minimal-memory loading** — reads each TSV with plain `csv`
   (no pandas), keeping only 3 cleaned strings per record
   (`name_clean`, `addr_clean`, `country_norm`) instead of full token
   lists/ngram sets.
2. **Lightweight blocking** — candidate generation blocks on
   `country + first-4-characters-of-cleaned-name` instead of a full
   trigram inverted index, with a per-bucket cap (`MAX_BUCKET`) so no single
   common prefix can stall the whole run.
3. **Word2Vec skipped** — the `w2v_cosine` feature is fixed at `0.0`; the
   other 12 hand-crafted similarity features (rapidfuzz ratios, token
   Jaccard, country match, length diffs) still drive the trained LightGBM
   classifier.
4. **Batched inference** — pairs are buffered and scored in batches of
   20,000 through `model.predict_proba()`, instead of one call per pair.
5. **Streaming output** — `candidate_pairs.tsv` is written row-by-row as
   candidates are generated, so partial progress is never lost in memory.

### Outputs
- `output/candidate_pairs.tsv` — blocking/candidate-generation output
  (`source1_entity_id`, `candidate_entity_ids`)
- `output/matching_results.tsv` — final matches
  (`source1_entity_id`, `matched_entity_ids`), covering all 1,732,544
  Source1 test entities

### Self-validation (on the training fold)
```
python -m src.score --pred output/train_matching_results.tsv --gt dataset/sample/train_ground_truth.tsv
```
```
Entities scored: 20000
Macro F0.5: 0.9230
```

### How to run
```powershell
cd amazon_ml__2027
python miti.py
```
Runs end-to-end on `dataset_original/test/` and writes both output files to
`output/`. Progress is printed every 300k rows loaded and every 10k S1
entities matched.

## Other scripts
- `fallback_submission.py` — emergency safety net used when the full run
  risked not finishing before the deadline. Builds a fully-valid
  `matching_results.tsv` (every Source1 entity present) from whatever
  `candidate_pairs.tsv` had been written so far, using the top-scored
  candidate as a best guess for entities already processed.
- `src/pipeline.py`, `src/preprocess.py`, `src/blocking.py`,
  `src/matching.py`, `src/embeddings.py`, `src/features.py` — the original
  modular training pipeline used to build and train the Word2Vec + LightGBM
  model (`models/word2vec_cbow.model`, `models/lightgbm_matcher.pkl`) on
  a smaller aligned sample (`make_aligned_sample.py`). `miti.py` reuses the
  saved LightGBM model from this pipeline for inference.

  #I didn't able to submit this solution within deadline as it takes too much time, this model journey is quite difficult.
  #So, yeah
             #I failed but I solved the problem!

## A note
Because of the memory crashes described above, the full-scale run only
finished after the Unstop submission portal had already closed for this
round — so unfortunately I never got to see this final version's actual
F0.5 score on the Amazon portal. The number above is a self-validated
score on the training fold, not the official test score.
