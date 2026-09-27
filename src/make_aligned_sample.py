"""
Ground-truth-ALIGNED sampler for train data.

Why this exists: `head -n 50000` on each source file independently breaks
alignment - a Source1 entity's true matches in Source2/Source3 can be
anywhere in those files, not necessarily in the first 50k lines. That's why
the earlier 50k slice had ~30 positives in it and the model learned nothing.

This script instead:
  1. Streams train_ground_truth.tsv (whole file, but it's tiny - just IDs).
  2. Randomly picks N1 Source1 entities that HAVE matches, and N2 Source1
     entities that are singletons (no match) - so the sample keeps a
     realistic mix (singletons matter for F0.5 too).
  3. Computes the exact set of Source2 / Source3 entity_ids that are the
     TRUE matches for the picked Source1 entities.
  4. Streams train_source1.tsv / train_source2.tsv / train_source3.tsv
     ONE LINE AT A TIME (csv.reader, not pandas.read_csv) and:
       - keeps a Source1 row iff its id is in the picked set
       - keeps a Source2/Source3 row iff its id is a required true-match id,
         OR randomly reservoir-samples it into a "distractor" pool (so
         blocking/candidate generation has real noise to search through,
         not just the answers).
  5. Writes the filtered train_source1/2/3.tsv + train_ground_truth.tsv
     (restricted to the picked Source1 ids) into --out-dir.

Memory use is O(sample size + distractor pool), never O(full file), so this
is safe on an 8GB machine even if the source files are 5M+ lines - each file
is read in a single streaming pass, never fully loaded.

Usage:
    python -m src.make_aligned_sample \
        --src-dir dataset_original \
        --out-dir dataset/train \
        --n-matched 15000 --n-singletons 5000 \
        --n-distractors-s2 40000 --n-distractors-s3 40000
"""
import argparse
import csv
import os
import random
import sys

csv.field_size_limit(sys.maxsize)


def load_ground_truth_ids(path):
    """Returns (matched_s1_ids: list[str], singleton_s1_ids: list[str],
    gt_map: dict[s1_id] -> list[matched_ids])."""
    matched, singleton, gt_map = [], [], {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            ids_str = row[1] if len(row) > 1 else ""
            ids = [x for x in ids_str.split(",") if x] if ids_str else []
            gt_map[s1_id] = ids
            (matched if ids else singleton).append(s1_id)
    return matched, singleton, gt_map


def reservoir_sample(pool_list, item, k, seen_count):
    """Classic single-pass reservoir sampling. `seen_count` is the count of
    eligible items seen so far *before* this one (0-indexed)."""
    if len(pool_list) < k:
        pool_list.append(item)
    else:
        j = random.randint(0, seen_count)
        if j < k:
            pool_list[j] = item


def stream_filter_source(src_path, out_path, keep_ids: set, distractor_k: int, rng_seed: int):
    """
    Single pass over a (possibly huge) source tsv:
      - writes every row whose entity_id is in keep_ids
      - additionally reservoir-samples up to distractor_k OTHER rows
        (uniform random, unbiased, memory O(distractor_k))
    Returns count of rows kept-as-match and kept-as-distractor.
    """
    random.seed(rng_seed)
    distractor_rows = []
    seen_others = 0
    kept_match = 0

    with open(src_path, newline="", encoding="utf-8") as f_in:
        reader = csv.reader(f_in, delimiter="\t")
        header = next(reader)
        with open(out_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            writer.writerow(header)
            for row in reader:
                if not row:
                    continue
                eid = row[0]
                if eid in keep_ids:
                    writer.writerow(row)
                    kept_match += 1
                elif distractor_k > 0:
                    reservoir_sample(distractor_rows, row, distractor_k, seen_others)
                    seen_others += 1
            for row in distractor_rows:
                writer.writerow(row)

    return kept_match, len(distractor_rows)


def stream_filter_source1(src_path, out_path, keep_ids: set):
    kept = 0
    with open(src_path, newline="", encoding="utf-8") as f_in:
        reader = csv.reader(f_in, delimiter="\t")
        header = next(reader)
        with open(out_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            writer.writerow(header)
            for row in reader:
                if row and row[0] in keep_ids:
                    writer.writerow(row)
                    kept += 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True,
                     help="Folder with the REAL full-size train_source1/2/3.tsv + train_ground_truth.tsv")
    ap.add_argument("--out-dir", required=True,
                     help="Where the aligned sample gets written (e.g. dataset/train)")
    ap.add_argument("--n-matched", type=int, default=15000,
                     help="How many Source1 entities WITH at least one match to include")
    ap.add_argument("--n-singletons", type=int, default=5000,
                     help="How many Source1 entities with NO match to include")
    ap.add_argument("--n-distractors-s2", type=int, default=40000,
                     help="Extra random (non-matching) Source2 rows to keep as noise for blocking")
    ap.add_argument("--n-distractors-s3", type=int, default=40000,
                     help="Extra random (non-matching) Source3 rows to keep as noise for blocking")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    gt_path = os.path.join(args.src_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(args.src_dir, "train_source1.tsv")
    s2_path = os.path.join(args.src_dir, "train_source2.tsv")
    s3_path = os.path.join(args.src_dir, "train_source3.tsv")

    print("Loading ground truth (id-only, small file)...")
    matched, singleton, gt_map = load_ground_truth_ids(gt_path)
    print(f"  total S1 entities: {len(gt_map)}  (matched={len(matched)}, singleton={len(singleton)})")

    n_matched = min(args.n_matched, len(matched))
    n_singleton = min(args.n_singletons, len(singleton))
    picked_matched = set(random.sample(matched, n_matched))
    picked_singleton = set(random.sample(singleton, n_singleton))
    picked_s1_ids = picked_matched | picked_singleton
    print(f"Picked {len(picked_matched)} matched + {len(picked_singleton)} singleton S1 entities "
          f"= {len(picked_s1_ids)} total")

    required_s2, required_s3 = set(), set()
    for s1_id in picked_matched:
        for mid in gt_map[s1_id]:
            if mid.startswith("S2-"):
                required_s2.add(mid)
            elif mid.startswith("S3-"):
                required_s3.add(mid)
    print(f"True matches needed: {len(required_s2)} Source2 ids, {len(required_s3)} Source3 ids")

    print("Streaming train_source1.tsv (single pass)...")
    kept_s1 = stream_filter_source1(s1_path, os.path.join(args.out_dir, "train_source1.tsv"), picked_s1_ids)
    print(f"  wrote {kept_s1} Source1 rows")

    print("Streaming train_source2.tsv (single pass, reservoir-sampling distractors)...")
    kept_m2, kept_d2 = stream_filter_source(
        s2_path, os.path.join(args.out_dir, "train_source2.tsv"),
        required_s2, args.n_distractors_s2, args.seed + 1)
    print(f"  wrote {kept_m2} true-match rows + {kept_d2} distractor rows")

    print("Streaming train_source3.tsv (single pass, reservoir-sampling distractors)...")
    kept_m3, kept_d3 = stream_filter_source(
        s3_path, os.path.join(args.out_dir, "train_source3.tsv"),
        required_s3, args.n_distractors_s3, args.seed + 2)
    print(f"  wrote {kept_m3} true-match rows + {kept_d3} distractor rows")

    gt_out_path = os.path.join(args.out_dir, "train_ground_truth.tsv")
    with open(gt_out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in picked_s1_ids:
            writer.writerow([s1_id, ",".join(gt_map[s1_id])])
    print(f"Wrote ground truth for {len(picked_s1_ids)} S1 entities -> {gt_out_path}")

    warn = []
    if kept_m2 < len(required_s2):
        warn.append(f"only found {kept_m2}/{len(required_s2)} required Source2 matches "
                     f"(some ids in ground truth may not exist in train_source2.tsv - check file integrity)")
    if kept_m3 < len(required_s3):
        warn.append(f"only found {kept_m3}/{len(required_s3)} required Source3 matches")
    if warn:
        print("\nWARNINGS:")
        for w in warn:
            print("  -", w)

    print("\nDONE. Sanity check counts above should roughly match: "
          "kept_m2/kept_m3 ~= required_s2/required_s3 (near 100%, since these ids MUST exist per the dataset spec).")


if __name__ == "__main__":
    main()
