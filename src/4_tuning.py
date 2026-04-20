import pandas as pd
import numpy as np
import os
import json
import warnings
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report

# LightGBM ve XGBoost'un gereksiz uyarılarını gizlemek için
warnings.filterwarnings("ignore")

def load_data_and_features(train_path, val_path, features_path, label_column='Label'):
    """
    Eğitim ve Validation verilerini yükler.
    Sadece bir önceki adımda seçilen en önemli özellikleri filtreler.
    """
    print(f"Eğitim verisi yükleniyor: {train_path}")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    with open(features_path, 'r', encoding='utf-8') as f:
        selected_features = json.load(f)
        
    X_train = df_train[selected_features]
    y_train = df_train[label_column]
    
    X_val = df_val[selected_features]
    y_val = df_val[label_column]
    
    return X_train, y_train, X_val, y_val

def calculate_healthy_smote_strategy(y):
    """
    Azınlık sınıfını kendi boyutunun en fazla 15 katına çıkaran ve 
    asla çoğunluk sınıfının %10'unu geçmesine izin vermeyen GÜVENLİ cap hesabı.
    """
    class_counts = y.value_counts()
    majority_count = class_counts.max()
    
    strategy = {}
    print("\n--- Sağlıklı SMOTE Stratejisi Hesaplanıyor ---")
    
    for cls, count in class_counts.items():
        if count == majority_count:
            strategy[cls] = count 
        else:
            healthy_target = int(min(count * 53, majority_count * 0.10))
            strategy[cls] = max(count, healthy_target)
            print(f"Sınıf {cls}: {count} adet -> Hedeflenen Sentetik Limit: {strategy[cls]} adet")
            
    return strategy

def apply_temporary_smote(X_train, y_train):
    """
    Aşırı öğrenmeyi engelleyen strateji ile SADECE eğitim verisine geçici SMOTE uygular.
    """
    print("\n--- Geçici SMOTE Uygulanıyor ---")
    healthy_strategy = calculate_healthy_smote_strategy(y_train)
    
    smote = SMOTE(sampling_strategy=healthy_strategy, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("\nSMOTE Sonrası Eğitim Seti Sınıf Dağılımı:")
    print(y_train_smote.value_counts().to_dict())
    
    return X_train_smote, y_train_smote

def optimize_dtypes_for_trees(X):
    """
    Ağaç algoritmalarının Port numaralarını yanlış yorumlamaması için kategorik tipe dönüştürür.
    """
    X_opt = X.copy()
    categorical_candidates = ['Dst Port', 'Src Port']
    
    for col in categorical_candidates:
        if col in X_opt.columns:
            X_opt[col] = X_opt[col].astype('category')
            
    return X_opt

def tune_hyperparameters(X_train, y_train):
    """
    XGBoost ve LightGBM için optimize edilmiş hiperparametre araması.
    GPU darbogazı çözüldü, terminal ilerleme çıktısı (verbose) eklendi.
    """
    print("\n--- Hiperparametre Optimizasyonu Başlıyor (Tuning) ---")
    print("Donanım: GPU (CUDA) Aktif Edildi!")
    
    # 1. XGBoost Optimizasyonu
    xgb_base = XGBClassifier(
        random_state=42, 
        eval_metric='mlogloss',
        tree_method='hist',   
        device='cuda',         
        enable_categorical=True
    )
    
    # Kapsam daraltıldı (Çok ağır modeller elendi)
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    
    print("\n[>>> XGBoost Eğitimi Başlıyor <<<]")
    print("Toplam 8 farklı kombinasyon, 3 katlamalı Çapraz Doğrulama (CV) ile denenecek (Toplam 24 Model).")
    
    # ÇÖZÜM: n_jobs=1 yapıldı (GPU tek başına çalışsın), verbose=3 yapıldı (terminalde görelim), n_iter=8'e düşürüldü
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_param_grid, n_iter=8, scoring='f1_macro', 
        cv=3, random_state=42, n_jobs=1, verbose=3
    )
    xgb_search.fit(X_train, y_train)
    
    # 2. LightGBM Optimizasyonu (GPU Hatası Giderildi)
    lgb_base = LGBMClassifier(
        random_state=42,
        n_jobs=-1            # CPU'nun tüm gücünü kullanması için ekle
    )
    
    lgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    
    print("\n[>>> LightGBM Eğitimi Başlıyor (CPU Modu) <<<]")
    lgb_search = RandomizedSearchCV(
        lgb_base, lgb_param_grid, n_iter=8, scoring='f1_macro', 
        cv=3, random_state=42, n_jobs=1, verbose=3
    )
    lgb_search.fit(X_train, y_train)
    
    return xgb_search.best_estimator_, xgb_search.best_params_, lgb_search.best_estimator_, lgb_search.best_params_

def evaluate_and_decide_architecture(xgb_model, lgb_model, X_val, y_val):
    """
    Eğitilen modelleri SAF Validation setinde yarıştırır.
    """
    print("\n--- Saf Validation Seti Üzerinde Test ve Mimari Kararı ---")
    
    y_pred_xgb = xgb_model.predict(X_val)
    y_pred_lgb = lgb_model.predict(X_val)
    
    xgb_f1 = f1_score(y_val, y_pred_xgb, average='macro')
    lgb_f1 = f1_score(y_val, y_pred_lgb, average='macro')
    
    print(f"\n=> XGBoost Validation F1-Score (Macro): %{xgb_f1*100:.2f}")
    print(f"=> LightGBM Validation F1-Score (Macro): %{lgb_f1*100:.2f}")
    
    score_diff = abs(xgb_f1 - lgb_f1)
    architecture_decision = {}
    
    if score_diff > 0.02:
        winner = "XGBoost" if xgb_f1 > lgb_f1 else "LightGBM"
        print(f"\nKarar (Solo): {winner} açık ara önde. Zayıf model elendi.")
        architecture_decision['strategy'] = 'Solo'
        architecture_decision['best_model'] = winner
    else:
        print("\nKarar (Hibrit - Soft Voting): Modeller başa baş! İkisi de tutulacak.")
        architecture_decision['strategy'] = 'Voting'
        architecture_decision['models'] = ['XGBoost', 'LightGBM']
        
    return architecture_decision

def save_recipe(architecture_decision, xgb_params, lgb_params, output_dir):
    """Optimum parametreleri JSON olarak kaydeder."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'params.json')
    
    recipe = {
        "architecture": architecture_decision,
        "parameters": {
            "XGBoost": xgb_params,
            "LightGBM": lgb_params
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(recipe, f, ensure_ascii=False, indent=4)
        
    print(f"\nTarifin sırrı (Parametreler) kaydedildi: {output_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    train_path = os.path.join(base_dir, "data", "processed", "X_train_sampled.csv")
    val_path = os.path.join(base_dir, "data", "processed", "X_val_ham.csv")
    features_path = os.path.join(base_dir, "configs", "selected_features.json")
    configs_dir = os.path.join(base_dir, "configs")
    
    print("=== 04_tuning.py (Hızlandırılmış Optimizasyon) Başlıyor ===")
    
    X_train, y_train, X_val, y_val = load_data_and_features(train_path, val_path, features_path)
    
    X_train_smote, y_train_smote = apply_temporary_smote(X_train, y_train)
    
    print("\n--- Veri Tipleri Ağaç Algoritmaları İçin Optimize Ediliyor ---")
    X_train_smote = optimize_dtypes_for_trees(X_train_smote)
    X_val = optimize_dtypes_for_trees(X_val)
    
    xgb_best, xgb_params, lgb_best, lgb_params = tune_hyperparameters(X_train_smote, y_train_smote)
    
    architecture_decision = evaluate_and_decide_architecture(xgb_best, lgb_best, X_val, y_val)
    save_recipe(architecture_decision, xgb_params, lgb_params, configs_dir)
    
    print("=== Optimizasyon Başarıyla Tamamlandı ===")

if __name__ == "__main__":
    main()