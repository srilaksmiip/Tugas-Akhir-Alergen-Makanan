# File: appStreamlit.py
import streamlit as st
import os
import sys
import joblib
import json
import pandas as pd
import ast
from googletrans import Translator

st.set_page_config(page_title="Modul Deteksi Alergen", layout="wide", initial_sidebar_state="expanded")

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_FOLDER_PATH = os.path.join(CURRENT_SCRIPT_DIR, 'src')
if SRC_FOLDER_PATH not in sys.path: sys.path.insert(0, SRC_FOLDER_PATH)

try:
    from config import MODEL_SAVE_DIR, TARGET_ALLERGENS_LIST, PROJECT_ROOT
    from dataPreprocessing import clean_ingredient_text
except ImportError as e: st.error(f"Kritis: Gagal mengimpor modul dari 'src': {e}.")

if 'PROJECT_ROOT' not in globals(): PROJECT_ROOT = CURRENT_SCRIPT_DIR 

PATH_KAMUS_JSON = os.path.join(PROJECT_ROOT, "data", "01-dataMentah", "kamusBahan.json")
PATH_MENU_KANTIN_DB = os.path.join(PROJECT_ROOT, "data", "01-dataMentah", "menuKantin.csv")
PATH_KAGGLE_DB = os.path.join(PROJECT_ROOT, "data", "02-intermediateOutput", "kaggle_all_processed_ingredients.csv")
MODEL_PREFIX_TO_LOAD = "RF"

HAPTEN_KEYWORDS = ["minyak", "lemak", "goreng", "gorengan"]

@st.cache_resource
def load_app_resources():
    _model, _kamus, _model_name = None, None, None
    _menu_db, _kaggle_db = pd.DataFrame(), pd.DataFrame()
    _errors = []
    try:
        with open(PATH_KAMUS_JSON, 'r', encoding='utf-8') as f: _kamus = json.load(f)
    except Exception as e: _errors.append(f"Gagal memuat kamus: {e}")
    try:
        model_dir_path = globals().get('MODEL_SAVE_DIR', os.path.join(PROJECT_ROOT, "models"))
        model_files = [f for f in os.listdir(model_dir_path) if f.startswith(f"{MODEL_PREFIX_TO_LOAD}_allergen_model")]
        if not model_files: _errors.append("Tidak ada file model RF yang ditemukan.")
        else:
            latest_model_path = os.path.join(model_dir_path, sorted(model_files)[-1])
            _model = joblib.load(latest_model_path)
            _model_name = sorted(model_files)[-1]
    except Exception as e: _errors.append(f"Gagal memuat model ML: {e}")
    
    try:
        _menu_db = pd.read_csv(PATH_MENU_KANTIN_DB, sep=',', encoding='utf-8-sig')
        if not all(col in _menu_db.columns for col in ['nama_makanan', 'daftar_bahan']): _menu_db = pd.DataFrame()
    except Exception as e: _errors.append(f"Gagal memuat menuKantin.csv: {e}")

    try:
        _kaggle_db = pd.read_csv(PATH_KAGGLE_DB)
        if not all(col in _kaggle_db.columns for col in ['Title', 'Ingredients']): _kaggle_db = pd.DataFrame()
    except Exception as e: _errors.append(f"Gagal memuat kaggle_recipes.csv: {e}")
        
    return _model, _kamus, _model_name, _menu_db, _kaggle_db, "\n".join(_errors) if _errors else None

model_pipeline, kamus_bahan, loaded_model_name, menu_kantin_df, kaggle_df, resource_load_error_msg = load_app_resources()

# --- Fungsi Logika Inti ---
@st.cache_resource
def get_translator(): return Translator()
translator = get_translator()

def process_ingredients_hybrid(ingredients_list_id, kamus_aktif, model_aktif):
    if not kamus_aktif or not model_aktif: return {"error_msg": "Kamus atau Model tidak tersedia."}
    english_mapped_terms_display, english_mapped_terms_processing, unmapped_log = [], [], []
    for item in ingredients_list_id:
        item_lookup = item.lower().strip()
        mapped_term = kamus_aktif.get(item_lookup)
        if mapped_term is None:
            try:
                translation = translator.translate(item_lookup, src='id', dest='en')
                mapped_term = translation.text.lower()
            except Exception:
                mapped_term = item_lookup
                unmapped_log.append(item_lookup)
        english_mapped_terms_display.append(mapped_term)
        english_mapped_terms_processing.append(mapped_term)
    st.session_state.unmapped_warning_msg = f"Bahan gagal diproses: {', '.join(unmapped_log)}" if unmapped_log else None
    cleaned_terms = [clean_ingredient_text(term) for term in english_mapped_terms_processing if term.strip()]
    feature_string = " ".join(cleaned_terms)
    detected_allergens, error_msg = {}, None
    if not feature_string: error_msg = "Tidak ada fitur teks yang valid setelah diproses."
    else:
        try:
            pred_bin = model_pipeline.predict([feature_string])[0]
            detected_allergens = {TARGET_ALLERGENS_LIST[i]: True for i, p in enumerate(pred_bin) if p == 1}
        except Exception as e: error_msg = f"Error saat prediksi: {e}"
    return {"mapped_str": ", ".join(english_mapped_terms_display), "cleaned_str": feature_string, "detected_allergens": detected_allergens, "error_msg": error_msg}

def search_recipe_in_db(query, db):
    if db.empty: return None, "Database resep publik tidak tersedia."
    query_lower = query.lower().strip()
    db['search_score'] = db['Title'].str.lower().apply(lambda title: 10 if query_lower == str(title) else (5 if query_lower in str(title) else str(title).count(query_lower)))
    best_match = db.sort_values('search_score', ascending=False).iloc[0]
    if best_match['search_score'] > 0:
        try: ingredients_list = ast.literal_eval(best_match['Ingredients'])
        except: ingredients_list = [ing.strip() for ing in str(best_match['Ingredients']).split(',')]
        return ingredients_list, best_match['Title']
    return None, f"Tidak ditemukan resep referensi untuk '{query}'."

# --- UI STREAMLIT ---
st.title("🥪 Modul Deteksi Risiko Alergen")
st.markdown("Prototipe ini mendemonstrasikan kapabilitas **modul deteksi alergen** dalam ekosistem **Smart Canteen**.")

# Sidebar
st.sidebar.header("Status Modul")
if model_pipeline: st.sidebar.success(f"Model ML: {loaded_model_name}")
else: st.sidebar.error("Model ML GAGAL dimuat.")
if kamus_bahan: st.sidebar.success(f"Kamus Bahan: {len(kamus_bahan)} entri.")
else: st.sidebar.error("Kamus Bahan GAGAL dimuat.")
if not menu_kantin_df.empty: st.sidebar.info(f"DB Menu Kantin: {len(menu_kantin_df)} menu.")
else: st.sidebar.warning("DB Menu Kantin GAGAL dimuat.")
if not kaggle_df.empty: st.sidebar.info(f"DB Resep Publik: {len(kaggle_df)} resep.")
else: st.sidebar.warning("DB Resep Publik GAGAL dimuat.")
st.sidebar.header("Filter Alergi Pribadi")
user_allergies = st.sidebar.multiselect("Pilih alergen yang ingin Anda hindari:", options=TARGET_ALLERGENS_LIST)

if resource_load_error_msg: st.error(f"Terjadi masalah saat inisialisasi:\n{resource_load_error_msg}")

if model_pipeline and kamus_bahan:
    # Inisialisasi session state
    if "analysis_result" not in st.session_state: st.session_state.analysis_result = {}
    if "step2_validation_data" not in st.session_state: st.session_state.step2_validation_data = None

    tab1, tab2 = st.tabs(["**🔬 Validasi Model (Uji Menu Kantin)**", "**🧑‍🍳 Simulasi Alur Kerja Tenant**"])

    # --- TAB 1: VALIDASI MODEL (Tetap Sama) ---
    with tab1:
        st.header("Validasi Kinerja Model pada Data Terkurasi")
        st.write("Menguji model pada data `menuKantin.csv` untuk memvalidasi performa dasarnya.")
        if not menu_kantin_df.empty:
            food_options = ["- Pilih Makanan -"] + menu_kantin_df['nama_makanan'].tolist()
            selected_food = st.selectbox("Pilih Makanan dari `menuKantin.csv`:", food_options, key="sb_kantin")
            if st.button("Validasi Model", type="primary", use_container_width=True, key="btn_tab1"):
                if selected_food != "- Pilih Makanan -":
                    with st.spinner(f"Menganalisis '{selected_food}'..."):
                        food_row = menu_kantin_df[menu_kantin_df['nama_makanan'] == selected_food].iloc[0]
                        ingredients_str = food_row['daftar_bahan']
                        ingredients_list = [ing.strip() for ing in ingredients_str.split(',')]
                        result = process_ingredients_hybrid(ingredients_list, kamus_bahan, model_pipeline)
                        st.session_state.analysis_result = {"food_name_display": selected_food, "original_ingredients": ingredients_str, **result}
                        st.session_state.step2_validation_data = None
                else: st.warning("Silakan pilih makanan.")
        else: st.warning("Database Menu Kantin tidak tersedia.")

    # --- TAB 2: SIMULASI TENANT (ALUR BARU YANG LEBIH CERDAS) ---
    with tab2:
        st.header("Simulasi: Tenant Menambahkan Menu Baru")
        
        # --- LANGKAH 1: PENGISIAN FORM UTAMA ---
        st.subheader("Langkah 1: Masukkan Detail Menu")
        with st.form(key="tenant_form"):
            col_form1, col_form2 = st.columns(2)
            with col_form1:
                tenant_food_name = st.text_input("Nama Makanan:", placeholder="Nasi Goreng Spesial")
                harga = st.number_input("Harga (Rp):", min_value=0, step=1000)
            with col_form2:
                st.file_uploader("Unggah Foto Makanan (Placeholder)", type=['jpg', 'png'])
            
            tenant_ingredients_initial = st.text_area("Daftar Bahan (Opsional):", placeholder="Jika sudah memiliki daftar bahan, silakan masukkan di sini. Jika kosong, sistem akan membantu mencari referensi.", height=100)
            submit_button = st.form_submit_button(label="Lanjutkan Proses", type="primary", use_container_width=True)

        if submit_button:
            st.session_state.analysis_result = {} # Reset hasil analisis
            st.session_state.step2_validation_data = None # Reset state validasi

            if not tenant_food_name.strip():
                st.warning("Nama Makanan wajib diisi.")
            else:
                # ALUR A: Tenant sudah mengisi bahan
                if tenant_ingredients_initial.strip():
                    with st.spinner("Menganalisis bahan yang diberikan..."):
                        ingredients_list_final = [b.strip() for b in tenant_ingredients_initial.split(',') if b.strip()]
                        result = process_ingredients_hybrid(ingredients_list_final, kamus_bahan, model_pipeline)
                        st.session_state.analysis_result = {
                            "food_name_display": tenant_food_name,
                            "original_ingredients": tenant_ingredients_initial,
                            **result
                        }
                # ALUR B: Tenant butuh bantuan dan masuk ke langkah validasi
                else:
                    with st.spinner(f"Mencari referensi untuk '{tenant_food_name}'..."):
                        ingredients_list, found_title = search_recipe_in_db(tenant_food_name, kaggle_df)
                        info_message = ""
                        draft_ingredients = ""
                        if ingredients_list:
                            draft_ingredients = ", ".join(ingredients_list)
                            info_message = f"💡 **Sistem menemukan resep referensi: '{found_title}'.** Mohon periksa dan sesuaikan daftar bahan di bawah ini."
                        else:
                            info_message = "⚠️ **Sistem tidak menemukan resep referensi.** Mohon masukkan daftar bahan secara manual."

                        st.session_state.step2_validation_data = {
                            "food_name": tenant_food_name,
                            "draft": draft_ingredients,
                            "info": info_message
                        }

        # --- LANGKAH VALIDASI ---
        if st.session_state.step2_validation_data:
            st.markdown("---")
            st.subheader("Langkah 2: Validasi Akhir Daftar Bahan")
            st.info(st.session_state.step2_validation_data["info"])

            validated_ingredients_str = st.text_area(
                "Daftar Bahan Final (WAJIB DIPERIKSA & DIEDIT):",
                value=st.session_state.step2_validation_data["draft"],
                height=150,
                key="final_ingredients_area"
            )

            if st.button("Simpan & Deteksi Alergen", type="primary", use_container_width=True, key="final_detect_btn"):
                if validated_ingredients_str.strip():
                    with st.spinner("Menganalisis..."):
                        ingredients_list_final = [b.strip() for b in validated_ingredients_str.split(',') if b.strip()]
                        result = process_ingredients_hybrid(ingredients_list_final, kamus_bahan, model_pipeline)
                        st.session_state.analysis_result = {
                            "food_name_display": st.session_state.step2_validation_data["food_name"],
                            "original_ingredients": validated_ingredients_str,
                            **result
                        }
                        st.session_state.step2_validation_data = None
                else:
                    st.warning("Daftar bahan tidak boleh kosong untuk dideteksi.")


    # --- AREA HASIL ---
    st.markdown("---")
    st.header("🔍 Hasil Analisis")

    result_data = st.session_state.get("analysis_result", {})

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.subheader("Input & Proses")
        if result_data:
            st.markdown(f"**Makanan:** `{result_data.get('food_name_display', 'N/A')}`")
            st.markdown("**Bahan Asli (ID):**"); st.code(result_data.get('original_ingredients', ''), language=None)
            st.markdown("**Bahan Setelah Mapping (EN):**"); st.code(result_data.get('mapped_str', ''), language=None)
            st.markdown("**String Fitur untuk Model:**"); st.code(result_data.get('cleaned_str', ''), language=None)
        else: st.info("Hasil akan ditampilkan di sini.")

    with col_res2:
        st.subheader("Output Prediksi & Personalisasi")
        if result_data.get("error_msg"): st.error(result_data["error_msg"])
        elif "detected_allergens" in result_data:
            detected = result_data["detected_allergens"]
            has_user_allergen = any(allergen in user_allergies for allergen in detected.keys())
            if user_allergies and has_user_allergen:
                st.error("🚨 **PERINGATAN!** Makanan ini mengandung alergen yang Anda hindari.")
            if detected:
                st.markdown("#### Potensi Alergen Terdeteksi:")
                for allergen_key in detected:
                    allergen_display_name = allergen_key.replace('_', ' ').capitalize()
                    if allergen_key in user_allergies: st.warning(f"⚠️ **{allergen_display_name}**")
                    else: st.success(f"✔️ {allergen_display_name}")
            else:
                st.success("✅ Tidak ada dari 14 alergen utama yang terdeteksi.")
                if user_allergies: st.info("Menu ini aman berdasarkan profil alergi Anda.")

            original_ingredients_str = result_data.get("original_ingredients", "").lower()

            found_keywords = [keyword for keyword in HAPTEN_KEYWORDS if keyword in original_ingredients_str]

            if found_keywords:
                st.markdown("---")
                st.subheader("📝 Catatan Risiko Tambahan")
                st.info(
                    f"""
                    **Potensi Risiko dari Lemak/Minyak (Hapten):**
                    Perlu diketahui, lemak atau minyak dapat berikatan dengan protein yang bepotensi menginduksi respon imun mirip alergi pada individu tertentu.
                    """
                )
    
    if st.session_state.get("unmapped_warning_msg"): st.sidebar.warning(st.session_state.unmapped_warning_msg)

st.caption("Prototipe Deteksi Alergen | Tugas Akhir Sistem dan Teknologi Informasi | Sri Laksmi Purwoningtyas | Institut Teknologi Bandung")
st.caption("Disclaimer: Hasil prediksi ini adalah output dari model machine learning dan berfungsi sebagai alat bantu informasi.")