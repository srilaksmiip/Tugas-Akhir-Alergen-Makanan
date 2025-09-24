# File: src/helper.py
import joblib
import os
import json
from datetime import datetime
# Impor path dari config.py yang ada di direktori yang sama (src)
from config import MODEL_SAVE_DIR, PARAM_SAVE_DIR, METRICS_SAVE_DIR

def save_model_to_disk(model_pipeline, model_name_prefix, timestamp=True, save_dir=MODEL_SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""
    filename = f"{model_name_prefix}_{time_str}.joblib" if time_str else f"{model_name_prefix}.joblib"
    filepath = os.path.join(save_dir, filename)
    joblib.dump(model_pipeline, filepath)
    print(f"Model disimpan: {filepath}")
    return filepath

def load_model_from_disk(model_filepath):
    try:
        model_pipeline = joblib.load(model_filepath)
        print(f"Model dimuat: {model_filepath}")
        return model_pipeline
    except FileNotFoundError:
        print(f"Error: File model tidak ditemukan di {model_filepath}")
        return None

def save_json_to_disk(data_dict, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=4)
    print(f"JSON disimpan: {filepath}")

def save_evaluation_metrics(metrics_collection, model_name, save_dir=METRICS_SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"metrics_report_{model_name}.txt")
    with open(filepath, 'w') as f:
        for metric_name, metric_value in metrics_collection.items():
            if isinstance(metric_value, dict): # Untuk classification report dict
                f.write(f"\n{metric_name}:\n")
                for class_label, scores in metric_value.items():
                    if isinstance(scores, dict):
                         f.write(f"  {class_label}:\n")
                         for score_name, score_val in scores.items():
                             f.write(f"    {score_name}: {score_val:.4f}\n")
                    else: # Untuk rata-rata seperti accuracy, macro avg, weighted avg
                        f.write(f"  {class_label}: {scores:.4f}\n")

            else:
                f.write(f"{metric_name}: {metric_value}\n")
    print(f"Metrik evaluasi disimpan: {filepath}")