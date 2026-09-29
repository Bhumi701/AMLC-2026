from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"

SAMPLE_DIR = DATA_DIR / "sample"        # jahan make_aligned_sample.py output likhta hai

TRAIN_S1 = SAMPLE_DIR / "train_source1.tsv"
TRAIN_S2 = SAMPLE_DIR / "train_source2.tsv"
TRAIN_S3 = SAMPLE_DIR / "train_source3.tsv"
TRAIN_GT = SAMPLE_DIR / "train_ground_truth.tsv"

# REAL test - final submission ke liye FULL test chahiye, 50k slice nahi.
# NOTE: apne actual folder ka naam yahan bilkul EXACT match karo
# (Explorer me check karo "dataset_original" hai ya "dataset_orignal" - typo se
#  FileNotFoundError aata hai training ke bilkul END me, sabse zyada waste hone wali jagah pe)
TEST_ORIGINAL_DIR = ROOT / "dataset_original" / "test"
TEST_S1 = TEST_ORIGINAL_DIR / "test_source1.tsv"
TEST_S2 = TEST_ORIGINAL_DIR / "test_source2.tsv"
TEST_S3 = TEST_ORIGINAL_DIR / "test_source3.tsv"

MAX_CANDIDATES_PER_ENTITY = 15     # 25 se 15 kiya - kam candidates = tez blocking + candidate-set-size ranking bonus
BLOCK_NGRAM = 3
COUNTRY_MUST_MATCH = True

W2V_VECTOR_SIZE = 80
W2V_WINDOW = 4
W2V_MIN_COUNT = 1
W2V_EPOCHS = 8                     # 12 se 8 kiya - time bachane ke liye, quality par bahut fark nahi padta
W2V_SG = 0          # CBOW
W2V_WORKERS = 2

SIMILARITY_THRESHOLD = 0.52
TOP_K_MATCHES = 4

OPTUNA_N_TRIALS = 15                # 20 se 15 kiya - time crunch me
OPTUNA_TIMEOUT = 180
SEED = 42
