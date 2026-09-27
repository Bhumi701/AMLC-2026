"""
Usage:
    python -m src.score --pred output/train_matching_results.tsv --gt dataset/sample/train_ground_truth.tsv
"""
import argparse
import pandas as pd


def f_beta(p, r, beta=0.5):
    if p == 0 and r == 0:
        return 0.0
    b2 = beta ** 2
    d = b2 * p + r
    return 0.0 if d == 0 else (1 + b2) * p * r / d


def score_entity(pred_ids, true_ids):
    if not true_ids and not pred_ids:
        return 1.0
    if not pred_ids:
        return 0.0
    tp = len(pred_ids & true_ids)
    p = tp / len(pred_ids)
    r = tp / len(true_ids) if true_ids else 0.0
    return f_beta(p, r)


def load_map(path, col):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    m = {}
    for _, row in df.iterrows():
        ids = row[col].split(",") if row[col] else []
        m[row["source1_entity_id"]] = set(i for i in ids if i)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()

    pred_map = load_map(args.pred, "matched_entity_ids")
    gt_map = load_map(args.gt, "matched_entity_ids")

    scores = [score_entity(pred_map.get(sid, set()), true_ids) for sid, true_ids in gt_map.items()]
    print(f"Entities scored: {len(scores)}")
    print(f"Macro F0.5: {sum(scores) / len(scores):.4f}")


if __name__ == "__main__":
    main()
