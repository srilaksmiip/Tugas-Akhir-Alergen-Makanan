# File: src/modelTraining.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import ( classification_report, hamming_loss, f1_score, roc_auc_score, accuracy_score, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from config import (
    CLEANED_TEXT_FEATURE_COL, TARGET_ALLERGENS_LIST, TRAIN_TEST_SPLIT_SIZE, GLOBAL_RANDOM_STATE,
    TFIDF_N_FEATURES, TFIDF_NGRAM_WINDOW, GRIDSEARCH_CV_FOLDS, FIGURES_SAVE_DIR,
    MODELS_FOR_TRAINING, METRICS_SAVE_DIR, PARAM_SAVE_DIR
)
from helper import save_model_to_disk, save_evaluation_metrics, save_json_to_disk

def plot_allergen_confusion_matrices(y_true_df, y_pred_array, target_labels, model_name_id, save_dir=FIGURES_SAVE_DIR):
    from sklearn.metrics import confusion_matrix as skl_cm
    os.makedirs(save_dir, exist_ok=True)
    num_targets = len(target_labels)
    grid_cols = 4
    grid_rows = (num_targets + grid_cols - 1) // grid_cols
    
    fig, all_axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 4.5 * grid_rows))
    all_axes = all_axes.flatten()

    for idx, current_label in enumerate(target_labels):
        if idx < len(all_axes): 
            ax_current = all_axes[idx]
            true_values = y_true_df[current_label] if isinstance(y_true_df, pd.DataFrame) and current_label in y_true_df.columns else y_true_df.iloc[:, idx]
            pred_values = y_pred_array[:, idx]
            
            cm_data = skl_cm(true_values, pred_values, labels=[0, 1])
            sns.heatmap(cm_data, annot=True, fmt='d', ax=ax_current, cmap='Blues', cbar=False,
                        xticklabels=['Non-Allergen', 'Allergen'], yticklabels=['Non-Allergen', 'Allergen'])
            ax_current.set_title(f'{current_label[:15]}', fontsize=10) 
            ax_current.set_xlabel('Predicted', fontsize=8)
            ax_current.set_ylabel('True', fontsize=8)
            ax_current.tick_params(axis='both', which='major', labelsize=8)
        else: 
            all_axes[idx].set_visible(False)

    for i in range(num_targets, len(all_axes)):
        all_axes[i].set_visible(False)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
    fig.suptitle(f'Confusion Matrices: {model_name_id}', fontsize=16)
    figure_path = os.path.join(save_dir, f'cm_plot_{model_name_id}.png')
    plt.savefig(figure_path)
    print(f"Plot CM disimpan: {figure_path}")
    plt.close(fig)

def plot_allergen_roc_curves(y_true_df, y_scores_or_probas_array, target_labels, model_name_id, save_dir=FIGURES_SAVE_DIR):
    """
    Memplot kurva ROC untuk setiap label alergen.
    y_scores_or_probas_array: Array NumPy dengan shape (n_samples, n_labels) berisi
                              skor positif atau probabilitas kelas positif.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_targets = len(target_labels)
    grid_cols = 4 
    grid_rows = (num_targets + grid_cols - 1) // grid_cols
    
    fig, all_axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 4.5 * grid_rows))
    all_axes = all_axes.flatten()

    for idx, current_label in enumerate(target_labels):
        if idx < len(all_axes): 
            ax_current = all_axes[idx]
            
            true_labels_for_allergen = y_true_df[current_label] if isinstance(y_true_df, pd.DataFrame) and current_label in y_true_df.columns else y_true_df.iloc[:, idx]
            scores_for_allergen = y_scores_or_probas_array[:, idx]
            if len(np.unique(true_labels_for_allergen)) > 1:
                fpr, tpr, _ = roc_curve(true_labels_for_allergen, scores_for_allergen)
                roc_auc = auc(fpr, tpr)

                ax_current.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
                ax_current.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_current.set_xlim([0.0, 1.0])
                ax_current.set_ylim([0.0, 1.05])
                ax_current.set_xlabel('False Positive Rate', fontsize=8)
                ax_current.set_ylabel('True Positive Rate', fontsize=8)
                ax_current.set_title(f'ROC: {current_label[:15]}', fontsize=10)
                ax_current.legend(loc="lower right", fontsize=7)
                ax_current.tick_params(axis='both', which='major', labelsize=8)
            else:
                ax_current.text(0.5, 0.5, 'Tidak bisa hitung ROC\n(hanya satu kelas di y_true)',
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=8, color='red')
                ax_current.set_title(f'ROC: {current_label[:15]} (Error)', fontsize=10)

        else: 
             all_axes[idx].set_visible(False)

    for i in range(num_targets, len(all_axes)):
        all_axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle(f'ROC Curves per Allergen - {model_name_id}', fontsize=16)
    figure_path = os.path.join(save_dir, f'roc_curves_{model_name_id}.png')
    plt.savefig(figure_path, bbox_inches='tight')
    print(f"Plot ROC Curves disimpan: {figure_path}")
    plt.close(fig)

def execute_model_training_and_evaluation(ml_ready_data_file_path):
    try:
        df_ready = pd.read_csv(ml_ready_data_file_path)
        print(f"Data ML Ready dimuat: {df_ready.shape}")
    except Exception as e:
        print(f"Error memuat data ML Ready dari {ml_ready_data_file_path}: {e}")
        return {}

    if df_ready.empty or CLEANED_TEXT_FEATURE_COL not in df_ready.columns:
        print("Data ML Ready kosong atau kolom fitur teks tidak ada. Training dibatalkan.")
        return {}

    X_features = df_ready[CLEANED_TEXT_FEATURE_COL].astype(str)
    actual_target_cols = [col for col in TARGET_ALLERGENS_LIST if col in df_ready.columns]
    if not actual_target_cols:
        print("Tidak ada kolom target alergen yang ditemukan di data. Training dibatalkan.")
        return {}
    print(f"Kolom target yang akan digunakan: {actual_target_cols}")
    y_targets = df_ready[actual_target_cols]


    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_targets, test_size=TRAIN_TEST_SPLIT_SIZE, random_state=GLOBAL_RANDOM_STATE, shuffle=True
    )

    tfidf_vec = TfidfVectorizer(
        max_features=TFIDF_N_FEATURES, ngram_range=TFIDF_NGRAM_WINDOW, stop_words='english'
    )

    model_definitions = {
        'LR': OneVsRestClassifier(LogisticRegression(random_state=GLOBAL_RANDOM_STATE, solver='liblinear', max_iter=1000)),
        'RF': OneVsRestClassifier(RandomForestClassifier(random_state=GLOBAL_RANDOM_STATE, n_jobs=-1)),
        'DT': OneVsRestClassifier(DecisionTreeClassifier(random_state=GLOBAL_RANDOM_STATE)),
        'LinearSVC': OneVsRestClassifier(LinearSVC(random_state=GLOBAL_RANDOM_STATE, dual=False, max_iter=2500)),
    }
    hyperparam_grids = {
        'LR': {'estimator__C': [0.1, 1, 10], 'estimator__class_weight': ['balanced', None]},
        'RF': {'estimator__n_estimators': [100, 150], 'estimator__max_depth': [None, 75],
               'estimator__min_samples_leaf': [1, 5], 'estimator__class_weight': ['balanced', None]},
        'DT': {'estimator__max_depth': [None, 70], 'estimator__min_samples_leaf': [1, 5, 10],
               'estimator__class_weight': ['balanced', None]},
        'LinearSVC': {'estimator__C': [0.1, 1, 10], 'estimator__class_weight': ['balanced', None],
                      'estimator__loss': ['squared_hinge']},
    }

    all_model_results = {}
    cv_strategy = KFold(n_splits=GRIDSEARCH_CV_FOLDS, shuffle=True, random_state=GLOBAL_RANDOM_STATE)

    for model_id in MODELS_FOR_TRAINING:
        if model_id not in model_definitions: continue
        print(f"\n--- Melatih Model: {model_id} ---")
        
        # Pipeline untuk TF-IDF dan Classifier
        training_pipeline = Pipeline([('tfidf', tfidf_vec), ('clf', model_definitions[model_id])])
        
        # Grid search parameter untuk classifier di dalam pipeline
        grid_params_for_pipeline = {f'clf__{key}': val for key, val in hyperparam_grids[model_id].items()}

        gs_cv = GridSearchCV(training_pipeline, param_grid=grid_params_for_pipeline, cv=cv_strategy,
                             scoring='f1_weighted', n_jobs=-1, verbose=1, return_train_score=True)
        gs_cv.fit(X_train, y_train)
        
        best_pipeline_found = gs_cv.best_estimator_
        print(f"Parameter terbaik {model_id}: {gs_cv.best_params_}")

        save_model_to_disk(best_pipeline_found, f"{model_id}_allergen_model")
        save_json_to_disk(gs_cv.best_params_, f"best_params_{model_id}.json", save_dir=PARAM_SAVE_DIR)

        y_pred_on_test = best_pipeline_found.predict(X_test)
        macro_mcc_test, macro_kappa_test = calculate_macro_averaged_metrics(y_test, y_pred_on_test)
        
        current_metrics = {
            "CV_Best_F1_Weighted": gs_cv.best_score_,
            "Test_F1_Weighted": f1_score(y_test, y_pred_on_test, average='weighted', zero_division=0),
            "Test_F1_Samples": f1_score(y_test, y_pred_on_test, average='samples', zero_division=0),
            "Test_F1_Macro": f1_score(y_test, y_pred_on_test, average='macro', zero_division=0),
            "Test_F1_Micro": f1_score(y_test, y_pred_on_test, average='micro', zero_division=0),
            "Test_Hamming_Loss": hamming_loss(y_test, y_pred_on_test),
            "Test_Subset_Accuracy": accuracy_score(y_test, y_pred_on_test),
            "Test_ROC_AUC_Weighted_OvR": "N/A",
            "Test_MCC_Macro": macro_mcc_test,
            "Test_Cohens_Kappa_Macro": macro_kappa_test,
            "Test_Classification_Report_Dict": classification_report(
                y_test, y_pred_on_test, target_names=actual_target_cols, output_dict=True, zero_division=0
            )
        }

        try:
            roc_auc_calculated = False
            if hasattr(best_pipeline_found, "predict_proba"):
                y_scores_or_probas_test = best_pipeline_found.predict_proba(X_test)
                if isinstance(y_scores_or_probas_test, list) and all(isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2 for arr in y_scores_or_probas_test):
                    positive_class_probas = np.array([proba_arr[:, 1] for proba_arr in y_scores_or_probas_test]).T
                    if y_test.shape == positive_class_probas.shape:
                        current_metrics["Test_ROC_AUC_Weighted_OvR"] = roc_auc_score(
                            y_test, positive_class_probas, average='weighted', multi_class='ovr'
                        )
                        roc_auc_calculated = True
                    else:
                        print(f"Shape mismatch untuk ROC AUC ({model_id}): y_test {y_test.shape}, probas {positive_class_probas.shape}")
                elif isinstance(y_scores_or_probas_test, np.ndarray) and y_scores_or_probas_test.ndim == 2:
                    if y_test.shape == y_scores_or_probas_test.shape:
                        current_metrics["Test_ROC_AUC_Weighted_OvR"] = roc_auc_score(
                            y_test, y_scores_or_probas_test, average='weighted', multi_class='ovr'
                        )
                        roc_auc_calculated = True
                    else:
                        print(f"Shape mismatch untuk ROC AUC ({model_id}): y_test {y_test.shape}, probas array {y_scores_or_probas_test.shape}")

            if not roc_auc_calculated and hasattr(best_pipeline_found.named_steps['clf'], "decision_function"):
                y_decision_vals_test = best_pipeline_found.decision_function(X_test)
                if y_test.shape == y_decision_vals_test.shape:
                    current_metrics["Test_ROC_AUC_Weighted_OvR"] = roc_auc_score(
                        y_test, y_decision_vals_test, average='weighted', multi_class='ovr'
                    )
                    roc_auc_calculated = True
                else:
                    print(f"Shape mismatch untuk ROC AUC (decision_function, {model_id}): y_test {y_test.shape}, decision_vals {y_decision_vals_test.shape}")
            
            if not roc_auc_calculated:
                print(f"Tidak bisa menghitung ROC AUC dengan metode standar untuk {model_id}. Output predict_proba/decision_function mungkin tidak sesuai.")

        except Exception as e:
            print(f"Error saat menghitung ROC AUC untuk {model_id}: {e}")

        
        y_scores_for_roc_plot = None
        roc_plot_possible = False
        try:
            if hasattr(best_pipeline_found, "predict_proba"):
                y_proba_test_list = best_pipeline_found.predict_proba(X_test) # Ini list of arrays
                if isinstance(y_proba_test_list, list) and \
                   all(isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2 for arr in y_proba_test_list):
                    y_scores_for_roc_plot = np.array([proba_arr[:, 1] for proba_arr in y_proba_test_list]).T
                    roc_plot_possible = True
                elif isinstance(y_proba_test_list, np.ndarray) and y_proba_test_list.ndim == 2: # Kasus fallback
                    y_scores_for_roc_plot = y_proba_test_list
                    roc_plot_possible = True
            elif hasattr(best_pipeline_found.named_steps['clf'], "decision_function"):
                y_scores_for_roc_plot = best_pipeline_found.decision_function(X_test)
                roc_plot_possible = True
            
            if roc_plot_possible and y_scores_for_roc_plot is not None:
                if y_test.shape == y_scores_for_roc_plot.shape:
                     plot_allergen_roc_curves(y_test, y_scores_for_roc_plot, actual_target_cols, model_id)
                else:
                    print(f"Shape mismatch, tidak bisa plot ROC untuk {model_id}: y_true {y_test.shape}, y_scores {y_scores_for_roc_plot.shape}")
            else:
                print(f"Tidak bisa mendapatkan skor/probabilitas yang sesuai untuk plot ROC untuk model {model_id}.")
        except Exception as e_roc_plot:
            print(f"Error saat menyiapkan data untuk plot ROC ({model_id}): {e_roc_plot}")

        all_model_results[model_id] = current_metrics
        save_evaluation_metrics(current_metrics, model_id) # Simpan metrik ke file
        
        print(f"Skor F1 Tertimbang (Test) untuk {model_id}: {current_metrics['Test_F1_Weighted']:.4f}")
        plot_allergen_confusion_matrices(y_test, y_pred_on_test, actual_target_cols, model_id)

        
        print(f"MCC Macro (Test) for {model_id}: {macro_mcc_test:.4f}")
        print(f"Cohen's Kappa Macro (Test) for {model_id}: {macro_kappa_test:.4f}")

    print("\n--- Ringkasan Perbandingan Model ---")
    summary_results_df = pd.DataFrame(all_model_results).T
    cols_for_summary = ['CV_Best_F1_Weighted', 'Test_F1_Weighted', 'Test_Subset_Accuracy', 'Test_Hamming_Loss', 'Test_ROC_AUC_Weighted_OvR']
    existing_summary_cols = [col for col in cols_for_summary if col in summary_results_df.columns]
    print(summary_results_df[existing_summary_cols])
    summary_results_df.to_csv(os.path.join(METRICS_SAVE_DIR, "model_comparison_summary.csv"))

    return all_model_results


def calculate_macro_averaged_metrics(y_true_df, y_pred_array):
    """
    Menghitung MCC dan Cohen's Kappa rata-rata (macro) untuk masalah multi-label.
    """
    mcc_scores = []
    kappa_scores = []
    
    num_labels = y_true_df.shape[1]
    
    for i in range(num_labels):
        y_true_label = y_true_df.iloc[:, i]
        y_pred_label = y_pred_array[:, i]

        if len(pd.unique(y_true_label)) < 2:
            mcc_scores.append(np.nan)
            kappa_scores.append(np.nan)
            continue

        mcc_scores.append(matthews_corrcoef(y_true_label, y_pred_label))
        kappa_scores.append(cohen_kappa_score(y_true_label, y_pred_label))
        
    macro_mcc = np.nanmean(mcc_scores) if mcc_scores else 0.0
    macro_kappa = np.nanmean(kappa_scores) if kappa_scores else 0.0
    
    return macro_mcc, macro_kappa


        
