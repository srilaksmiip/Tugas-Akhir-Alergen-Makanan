# File: src/config.py
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Paths Data
RAW_OFF_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "01-dataMentah", "openfoodfacts.tsv")
CUSTOM_ALLERGENS_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "01-dataMentah", "turunanAlergen.csv")
INTERMEDIATE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "02-intermediateOutput")
PROCESSED_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "03-processedOutput")
KAGGLE_DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "data", "01-dataMentah", "kaggle")

MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
PARAM_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "param")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_SAVE_DIR = os.path.join(REPORTS_DIR, "figures")
METRICS_SAVE_DIR = os.path.join(REPORTS_DIR, "metrics")
ALL_KAGGLE_INGREDIENTS_SAVE_PATH = os.path.join(INTERMEDIATE_OUTPUT_DIR, "kaggle_all_processed_ingredients.csv") 
ALL_INGREDIENT_FREQUENCY_SAVE_PATH = os.path.join(INTERMEDIATE_OUTPUT_DIR, "kaggle_all_ingredient_frequencies.csv")

ML_READY_COMBINED_PATH = os.path.join(PROCESSED_OUTPUT_DIR, "data_ml_ready_combined.csv")
RAW_OFF_ALLERGEN_COLUMNS_ORIGINAL_NAMES = [
    "en:celery", "en:crustaceans", "en:eggs", "en:fish", "en:gluten",
    "en:lupin", "en:milk", "en:molluscs", "en:mustard", "en:nuts",
    "en:peanuts", "en:sesame-seeds", "en:soybeans", "en:sulphur-dioxide-and-sulphites"
]
RAW_COLUMNS_TO_KEEP_OFF = [
    "product_name",
    "ingredients_text"
] + RAW_OFF_ALLERGEN_COLUMNS_ORIGINAL_NAMES

# Nama bersih alergen target (setelah 'en:' dihapus dan '-' jadi '_')
TARGET_ALLERGENS_LIST = [
    "celery", "crustaceans", "eggs", "fish", "gluten", "lupin", "milk",
    "molluscs", "mustard", "nuts", "peanuts", "sesame_seeds", "soybeans",
    "sulphur_dioxide_and_sulphites"
]
ALLERGEN_TAG_PREFIX = "en:"

CLEANED_TEXT_FEATURE_COL = "ingredients_text_cleaned"

TRAIN_TEST_SPLIT_SIZE = 0.2
GLOBAL_RANDOM_STATE = 42
GRIDSEARCH_CV_FOLDS = 3
TFIDF_N_FEATURES = 1500
TFIDF_NGRAM_WINDOW = (1, 2)
MODELS_FOR_TRAINING = ["RF"]