# File: src/dataPreprocessing.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os


from config import (
    RAW_COLUMNS_TO_KEEP_OFF, CLEANED_TEXT_FEATURE_COL, TARGET_ALLERGENS_LIST,
    ALLERGEN_TAG_PREFIX, RAW_OFF_ALLERGEN_COLUMNS_ORIGINAL_NAMES
)

def download_nltk_resource_if_needed(resource_id, download_package_name=None):
    if download_package_name is None:
        download_package_name = resource_id.split('/')[-1]

    try:
        nltk.data.find(resource_id)
        print(f"Resource NLTK '{resource_id}' ('{download_package_name}') sudah ada.")
    except LookupError:
        print(f"Resource NLTK '{resource_id}' ('{download_package_name}') tidak ditemukan. Mengunduh...")
        try:
            nltk.download(download_package_name, quiet=False)
            print(f"Berhasil mengunduh '{download_package_name}'.")
        except AttributeError as ae:
            print(f"AttributeError saat mencoba mengunduh '{download_package_name}': {ae}")
            print("Ini mungkin karena versi NLTK atau masalah konfigurasi. Pastikan NLTK terinstal dengan benar.")
        except Exception as e:
            print(f"Gagal mengunduh '{download_package_name}': {e}")
            print("Pastikan Anda memiliki koneksi internet.")



nltk_resource_map = {
    'corpora/wordnet': 'wordnet',
    'corpora/stopwords': 'stopwords',
    'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
    'tokenizers/punkt': 'punkt',
    'corpora/omw-1.4': 'omw-1.4',
    'tokenizers/punkt_tab': 'punkt_tab', 
    'taggers/averaged_perceptron_tagger_eng': 'averaged_perceptron_tagger_eng' 
}
for resource_path_id, download_name in nltk_resource_map.items():
    download_nltk_resource_if_needed(resource_path_id, download_name)

lemmatizer = WordNetLemmatizer()
english_stop_words = set(stopwords.words('english'))

def get_wordnet_tag(word_token):
    # nltk.pos_tag menggunakan averaged_perceptron_tagger secara default
    nltk_tag = nltk.pos_tag([word_token])[0][1][0].upper()
    tag_map = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
    return tag_map.get(nltk_tag, wn.NOUN)

def clean_ingredient_text(raw_text):
    if not isinstance(raw_text, str): return ""
    text = raw_text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_tag(token)) for token in tokens]
    meaningful_tokens = [
        token for token in lemmatized_tokens
        if token not in english_stop_words and len(token) > 2
    ]
    return " ".join(meaningful_tokens)

def apply_text_cleaning_to_series(text_series):
    return text_series.apply(clean_ingredient_text)

def preprocess_open_food_facts(raw_off_file_path):
    try:
        df_off = None
        try:
            print("Mencoba memuat OpenFoodFacts dengan separator ';'")
            df_off = pd.read_csv(raw_off_file_path, sep=';', usecols=RAW_COLUMNS_TO_KEEP_OFF, low_memory=False, on_bad_lines='warn')
        except (pd.errors.ParserError, ValueError) as e1:
            print(f"Gagal dengan ';': {e1}. Mencoba separator TAB '\t' untuk OpenFoodFacts...")
            try:
                df_off = pd.read_csv(raw_off_file_path, sep='\t', usecols=RAW_COLUMNS_TO_KEEP_OFF, low_memory=False, on_bad_lines='warn')
            except Exception as e2:
                print(f"Gagal juga dengan '\t': {e2}")
                raise 
        
        print(f"Data mentah OpenFoodFacts dimuat: {df_off.shape}")
    except Exception as e:
        print(f"Error memuat OpenFoodFacts dari {raw_off_file_path}: {e}")
        return pd.DataFrame()

    if 'ingredients_text' not in df_off.columns:
        print("Error: Kolom 'ingredients_text' tidak ditemukan di OpenFoodFacts.")
        return pd.DataFrame()
        
    df_off[CLEANED_TEXT_FEATURE_COL] = apply_text_cleaning_to_series(df_off['ingredients_text'])
    
    rename_map_off = {}
    for original_name in RAW_OFF_ALLERGEN_COLUMNS_ORIGINAL_NAMES: 
        if original_name in df_off.columns:
            clean_name = original_name.replace(ALLERGEN_TAG_PREFIX, "").replace("-", "_")
            rename_map_off[original_name] = clean_name
    df_off.rename(columns=rename_map_off, inplace=True)

    final_cols_off = [CLEANED_TEXT_FEATURE_COL]
    for target_allergen in TARGET_ALLERGENS_LIST:
        if target_allergen not in df_off.columns:
            df_off[target_allergen] = 0
        final_cols_off.append(target_allergen)
    
    df_off_processed = df_off[final_cols_off].copy()
    df_off_processed.dropna(subset=[CLEANED_TEXT_FEATURE_COL], inplace=True)
    df_off_processed = df_off_processed[df_off_processed[CLEANED_TEXT_FEATURE_COL].str.strip().astype(bool)]
    print(f"Data OpenFoodFacts setelah diproses: {df_off_processed.shape}")
    return df_off_processed

def preprocess_turunan_alergen_data(turunan_alergen_file_path):
    try:
        df_turunan = pd.read_csv(turunan_alergen_file_path, sep='\t')
        print(f"Data 'turunanAlergen.csv' dimuat: {df_turunan.shape}")
    except Exception as e:
        print(f"Error memuat 'turunanAlergen.csv' dari {turunan_alergen_file_path}: {e}")
        return pd.DataFrame()

    if 'ingredients_text' not in df_turunan.columns:
        print("Error: Kolom 'ingredients_text' tidak ada di 'turunanAlergen.csv'.")
        return pd.DataFrame()
    
    df_turunan[CLEANED_TEXT_FEATURE_COL] = apply_text_cleaning_to_series(df_turunan['ingredients_text'])

    cols_to_rename_turunan = {}
    for col in df_turunan.columns:
        if col.startswith(ALLERGEN_TAG_PREFIX):
            clean_name = col.replace(ALLERGEN_TAG_PREFIX, "").replace("-", "_")
            if clean_name in TARGET_ALLERGENS_LIST:
                cols_to_rename_turunan[col] = clean_name
    df_turunan.rename(columns=cols_to_rename_turunan, inplace=True)

    final_cols_turunan = [CLEANED_TEXT_FEATURE_COL]
    for allergen_name in TARGET_ALLERGENS_LIST:
        if allergen_name not in df_turunan.columns:
            df_turunan[allergen_name] = 0
        final_cols_turunan.append(allergen_name)
    
    df_turunan_processed = df_turunan[final_cols_turunan].copy()
    df_turunan_processed.dropna(subset=[CLEANED_TEXT_FEATURE_COL], inplace=True)
    df_turunan_processed = df_turunan_processed[df_turunan_processed[CLEANED_TEXT_FEATURE_COL].str.strip().astype(bool)]
    print(f"Data 'turunanAlergen.csv' setelah diproses: {df_turunan_processed.shape}")
    return df_turunan_processed

def execute_full_preprocessing(raw_off_file, turunan_alergen_file, cleaned_off_text_save_path, ml_ready_combined_save_path):
    print("Memulai Proses Pra-Pemrosesan Data...")
    
    df_off_ml_ready = preprocess_open_food_facts(raw_off_file)
    if df_off_ml_ready.empty:
        print("Gagal memproses data OpenFoodFacts. Pipeline dihentikan.")
        return None
    
    print(f"Menyimpan data OFF yang sudah dibersihkan ke {cleaned_off_text_save_path}")
    try :
        os.makedirs(os.path.dirname(cleaned_off_text_save_path), exist_ok=True)
        df_off_ml_ready.to_csv(cleaned_off_text_save_path, index=False)
        print(f"Data OpenFoodFacts yang sudah dibersihkan disimpan: {cleaned_off_text_save_path}")
    except Exception as e:
        print(f"Error saat menyimpan data OpenFoodFacts yang sudah dibersihkan: {e}")
        return None

    df_turunan_ml_ready = preprocess_turunan_alergen_data(turunan_alergen_file)

    if not df_turunan_ml_ready.empty:
        all_target_cols_ordered = [CLEANED_TEXT_FEATURE_COL] + TARGET_ALLERGENS_LIST
        
        df_off_final_cols = []
        for col in all_target_cols_ordered:
            if col in df_off_ml_ready.columns:
                df_off_final_cols.append(col)
            elif col == CLEANED_TEXT_FEATURE_COL: 
                 print(f"KRITIS: Kolom {CLEANED_TEXT_FEATURE_COL} tidak ada di df_off_ml_ready")
                 return None 
            else: # Kolom target alergen
                df_off_ml_ready[col] = 0
                df_off_final_cols.append(col)
        df_off_final = df_off_ml_ready[df_off_final_cols].copy()


        df_turunan_final_cols = []
        for col in all_target_cols_ordered:
            if col in df_turunan_ml_ready.columns:
                df_turunan_final_cols.append(col)
            elif col == CLEANED_TEXT_FEATURE_COL:
                 print(f"KRITIS: Kolom {CLEANED_TEXT_FEATURE_COL} tidak ada di df_turunan_ml_ready")
                 return None
            else:
                df_turunan_ml_ready[col] = 0
                df_turunan_final_cols.append(col)
        df_turunan_final = df_turunan_ml_ready[df_turunan_final_cols].copy()
            
        df_combined_ml_ready = pd.concat(
            [df_off_final, df_turunan_final], 
            ignore_index=True,
        )
        print(f"Data OpenFoodFacts dan Turunan Alergen digabung: {df_combined_ml_ready.shape}")
    else:
        print("Data Turunan Alergen kosong, hanya menggunakan data OpenFoodFacts.")
        df_combined_ml_ready = df_off_ml_ready.copy() 

    final_ordered_cols = [CLEANED_TEXT_FEATURE_COL] + TARGET_ALLERGENS_LIST
    for col in final_ordered_cols:
        if col not in df_combined_ml_ready.columns:
             df_combined_ml_ready[col] = 0
    
    df_combined_ml_ready = df_combined_ml_ready[final_ordered_cols] # Reorder kolom

    os.makedirs(os.path.dirname(ml_ready_combined_save_path), exist_ok=True)
    df_combined_ml_ready.to_csv(ml_ready_combined_save_path, index=False)
    print(f"Data gabungan siap ML disimpan: {ml_ready_combined_save_path}")
    print("Pra-pemrosesan data selesai.")
    return ml_ready_combined_save_path