# File: src/plot_distributions.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, os.pardir)) 
if PROJECT_ROOT_DIR not in sys.path: 
    sys.path.insert(0, PROJECT_ROOT_DIR)

from src.config import (
    ML_READY_COMBINED_PATH,
    CUSTOM_ALLERGENS_DATA_PATH,
    CLEANED_OFF_TEXT_PATH,
    TARGET_ALLERGENS_LIST,
    FIGURES_SAVE_DIR,
)
from src.dataPreprocessing import preprocess_turunan_alergen_data 

def plot_allergen_distribution(df, target_allergen_columns, title_suffix, save_filename_suffix):
    """
    Membuat dan menyimpan plot distribusi untuk kolom-kolom alergen target.
    """
    if df is None or df.empty:
        print(f"DataFrame kosong atau None, tidak bisa membuat plot untuk: {title_suffix}")
        return

    existing_target_columns = [col for col in target_allergen_columns if col in df.columns]
    if not existing_target_columns:
        print(f"Tidak ada kolom target alergen yang ditemukan di DataFrame untuk: {title_suffix}")
        return

    allergen_counts = df[existing_target_columns].sum().sort_values(ascending=False)
    if allergen_counts.empty:
        print(f"Tidak ada data alergen untuk diplot pada {title_suffix}")
        return

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=allergen_counts.index, y=allergen_counts.values, palette="viridis")
    plt.xlabel("Jenis Alergen", fontsize=12)
    plt.ylabel("Jumlah Kemunculan (Sampel Positif)", fontsize=12)
    plt.title(f"Distribusi Frekuensi Alergen - {title_suffix}", fontsize=15, pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    max_val = allergen_counts.values.max() if len(allergen_counts.values) > 0 else 1 # Hindari error jika kosong
    for i, v in enumerate(allergen_counts.values):
        ax.text(i, v + (max_val * 0.01), str(int(v)), color='black', ha='center', va='bottom', fontsize=9) # Ubah ke int

    plt.tight_layout()
    
    os.makedirs(FIGURES_SAVE_DIR, exist_ok=True)
    full_save_path = os.path.join(FIGURES_SAVE_DIR, f"distribusi_alergen_{save_filename_suffix}.png")
    plt.savefig(full_save_path)
    print(f"Plot disimpan sebagai: {full_save_path}")
    plt.close()

def main_plotter():
    print("Memulai pembuatan plot distribusi alergen terpisah...")

    # 1. Plot untuk Dataset Open Food Facts Murni (Setelah Diproses)
    try:
        if os.path.exists(CLEANED_OFF_TEXT_PATH): # Path dari config.py
            df_off_only = pd.read_csv(CLEANED_OFF_TEXT_PATH)
            print(f"\nData Open Food Facts (saja) dimuat dari {CLEANED_OFF_TEXT_PATH}: {df_off_only.shape}")
            plot_allergen_distribution(
                df_off_only,
                TARGET_ALLERGENS_LIST,
                "Open Food Facts (Diproses)", 
                "openfoodfacts_saja"           
            )
        else:
            print(f"File tidak ditemukan: {CLEANED_OFF_TEXT_PATH}. Plot untuk Open Food Facts (saja) dilewati.")
    except Exception as e:
        print(f"Error saat memproses data Open Food Facts (saja): {e}")

    # 2. Plot untuk Dataset turunanAlergen.csv (Setelah Diproses)
    print(f"\nMemproses turunanAlergen.csv untuk plot distribusi...")
    df_turunan_processed = preprocess_turunan_alergen_data(CUSTOM_ALLERGENS_DATA_PATH)
    
    if df_turunan_processed is not None and not df_turunan_processed.empty:
        plot_allergen_distribution(
            df_turunan_processed,
            TARGET_ALLERGENS_LIST,
            "Turunan Alergen (Diproses)", # 
            "turunan_alergen"             
        )
    else:
        print("Gagal memproses atau data turunan alergen kosong, plot dilewati.")

    # 3. Plot untuk Dataset Gabungan (Open Food Facts + Turunan Alergen)
    try:
        if os.path.exists(ML_READY_COMBINED_PATH):
            df_combined = pd.read_csv(ML_READY_COMBINED_PATH)
            print(f"\nData gabungan dimuat dari {ML_READY_COMBINED_PATH}: {df_combined.shape}")
            plot_allergen_distribution(
                df_combined,
                TARGET_ALLERGENS_LIST,
                "Training Gabungan (OFF + Turunan)", 
                "training_gabungan"                  
            )
        else:
            print(f"File tidak ditemukan: {ML_READY_COMBINED_PATH}. Plot untuk data gabungan dilewati.")
    except Exception as e:
        print(f"Error saat memproses data gabungan: {e}")

    print("\nPembuatan plot distribusi alergen selesai.")

if __name__ == "__main__":
    main_plotter()