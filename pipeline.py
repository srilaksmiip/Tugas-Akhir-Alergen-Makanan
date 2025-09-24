# File: main_pipeline.py
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.config import (
    RAW_OFF_DATA_PATH, CUSTOM_ALLERGENS_DATA_PATH, CLEANED_OFF_TEXT_PATH, ML_READY_COMBINED_PATH
)
from src.dataPreprocessing import execute_full_preprocessing
from src.modelTraining import execute_model_training_and_evaluation

def run_complete_pipeline():
    print("Memulai Pipeline Klasifikasi Alergen...")

    # 1. Pra-pemrosesan Data
    final_ml_ready_file = execute_full_preprocessing(
        raw_off_file=RAW_OFF_DATA_PATH,
        turunan_alergen_file=CUSTOM_ALLERGENS_DATA_PATH,
        cleaned_off_text_save_path=CLEANED_OFF_TEXT_PATH,
        ml_ready_combined_save_path=ML_READY_COMBINED_PATH
    )

    if final_ml_ready_file and os.path.exists(final_ml_ready_file):
        print(f"\nData siap untuk ML telah disimpan di: {final_ml_ready_file}")

        # 2. Pelatihan dan Evaluasi Model
        print("\nMemulai Pelatihan dan Evaluasi Model...")
        training_results = execute_model_training_and_evaluation(final_ml_ready_file)

        if training_results:
            print("\n--- Ringkasan Hasil Akhir Pipeline ---")
            for model_name, metrics_data in training_results.items():
                print(f"\nModel: {model_name}")
                print(f"  Skor F1 Tertimbang CV Terbaik: {metrics_data.get('CV_Best_F1_Weighted', 'N/A'):.4f}")
                print(f"  F1 Tertimbang (Test): {metrics_data.get('Test_F1_Weighted', 'N/A'):.4f}")
                print(f"  Akurasi Subset (Test): {metrics_data.get('Test_Subset_Accuracy', 'N/A'):.4f}")
        else:
            print("Pelatihan model tidak menghasilkan output.")
    else:
        print("Pipeline dihentikan karena pra-pemrosesan data gagal atau file tidak ditemukan.")

    print("\nPipeline Klasifikasi Alergen Selesai.")

if __name__ == "__main__":
    run_complete_pipeline()