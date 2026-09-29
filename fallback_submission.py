"""
EMERGENCY FALLBACK - run in a SEPARATE terminal, do NOT stop the main
matching run. Reads whatever candidate_pairs.tsv has been written so far
(it's being written live, row by row) and builds a FULLY VALID
matching_results.tsv covering EVERY Source1 entity (required by the
validator), using the top-scored candidate as a guess for entities already
reached, and empty (singleton guess) for entities not reached yet.

This guarantees you have a submittable file even if the main run doesn't
finish in time. If the main run DOES finish, its matching_results.tsv is
better quality - upload that instead / in addition.
"""
import csv
from pathlib import Path

csv.field_size_limit(2**31 - 1)

ROOT = Path(__file__).resolve().parent
TEST_S1 = ROOT / "dataset_original" / "test" / "test_source1.tsv"
CAND_PATH = ROOT / "output" / "candidate_pairs.tsv"
OUT_PATH = ROOT / "output" / "matching_results_FALLBACK.tsv"


def main():
    print("Loading S1 ids...", flush=True)
    s1_ids = []
    with TEST_S1.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        idx = header.index("entity_id")
        for row in r:
            if row:
                s1_ids.append(row[idx])
    print(f"S1 total: {len(s1_ids)}", flush=True)

    print("Reading candidate_pairs.tsv snapshot (whatever progress exists so far)...", flush=True)
    best_guess = {}
    if CAND_PATH.is_file():
        with CAND_PATH.open(newline="", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            header = next(r, None)
            for row in r:
                if not row or len(row) < 2:
                    continue
                sid = row[0]
                cands = row[1]
                if cands:
                    best_guess[sid] = cands.split(",")[0]
    print(f"Rows with at least one candidate so far: {len(best_guess)}", flush=True)

    print("Writing fallback matching_results.tsv...", flush=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for sid in s1_ids:
            w.writerow([sid, best_guess.get(sid, "")])

    print("DONE ->", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()