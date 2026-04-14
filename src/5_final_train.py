import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore")

def load_and_combine_pure_data(train_path, val_path, features_path, label_column='Label'):
    """
    Kilitli kasadaki saf X_train_ham ve X_val_ham dosyalarını yükler.
    Geçici (sampled) veriyi çöpe atar. İki saf veriyi birleştirerek %85'lik dev eğitim setini oluşturur.
    Sadece seçilen en önemli özellikleri filtreler.
    """
    print(f"Saf Train Seti yükleniyor: {train_path}")
    print(f"Saf Validation Seti yükleniyor: {val_path}")
    
    df_train_pure = pd.read_csv(train_path)
    df_val_pure = pd.read_csv(val_path)
    
    # Dev seti oluştur (Train + Validation Birleşimi)
    df_dev = pd.concat([df_train_pure, df_val_pure], ignore_index=True)
    
    with open(features_path, 'r', encoding='utf-8') as f:
        selected_features = json.load(f)
        
    X_dev = df_dev[selected_features]
    y_dev = df_dev[label_column]
    
    print(f"\nDevasa Saf Eğitim Seti (Dev Set) oluşturuldu. Toplam Satır: {len(df_dev)}")
    return X_dev, y_dev

def calculate_healthy_smote_strategy(y):
    """
    Azınlık sınıfını kendi boyutunun en fazla 15 katına çıkaran ve 
    asla çoğunluk sınıfının %10'unu geçmesine izin vermeyen GÜVENLİ cap hesabı.
    """
    class_counts = y.value_counts()
    majority_count = class_counts.max()
    strategy = {}
    
    for cls, count in class_counts.items():
        if count == majority_count:
            strategy[cls] = count 
        else:
            healthy_target = int(min(count * 15, majority_count * 0.10))
            strategy[cls] = max(count, healthy_target)
            
    return strategy

def apply_final_smote(X_dev, y_dev):
    """
    Birleştirilmiş %85'lik dev sete sıfırdan ve TEK SEFERLİK nihai SMOTE uygular.
    """
    print("\n--- Nihai SMOTE Uygulanıyor (Dev Set Üzerine) ---")
    print("Orijinal Dev Set Sınıf Dağılımı:")
    print(y_dev.value_counts().to_dict())
    
    healthy_strategy = calculate_healthy_smote_strategy(y_dev)
    
    smote = SMOTE(sampling_strategy=healthy_strategy, random_state=42)
    X_dev_smote, y_dev_smote = smote.fit_resample(X_dev, y_dev)
    
    print("\nNihai SMOTE Sonrası Dev Set Sınıf Dağılımı (Kusursuzlaştırma):")
    print(y_dev_smote.value_counts().to_dict())
    
    return X_dev_smote, y_dev_smote

def optimize_dtypes_for_trees(X):
    """Ağaç algoritmalarının Port numaralarını yanlış yorumlamaması için optimize eder."""
    X_opt = X.copy()
    categorical_candidates = ['Dst Port', 'Src Port']
    for col in categorical_candidates:
        if col in X_opt.columns:
            X_opt[col] = X_opt[col].astype('category')
    return X_opt

def build_and_train_final_model(X_train, y_train, recipe_path):
    """
    4. Adımda kilitli kasaya kaydedilen 'Tarifin Sırrı' (params.json) dosyasını okur.
    Mimari kararına göre (Solo veya Hibrit) yepyeni bir model objesi yaratır ve SADECE EĞİTİR.
    """
    print(f"\n--- Nihai Model İnşası ve Eğitim Başlıyor ---")
    print(f"Tarifin Sırrı (Parametreler) Okunuyor: {recipe_path}")
    
    with open(recipe_path, 'r', encoding='utf-8') as f:
        recipe = json.load(f)
        
    architecture = recipe['architecture']['strategy']
    xgb_params = recipe['parameters']['XGBoost']
    lgb_params = recipe['parameters']['LightGBM']
    
    # Parametreleri algoritmaların anlayacağı temiz bir formata getir (Listeleri çıkar)
    xgb_params = {k: v for k, v in xgb_params.items()}
    lgb_params = {k: v for k, v in lgb_params.items()}
    
    # 1. XGBoost Objesi (Her halükarda hazırla)
    xgb_model = XGBClassifier(
        **xgb_params, 
        random_state=42, 
        eval_metric='mlogloss',
        tree_method='hist',
        device='cuda',
        enable_categorical=True
    )
    
    # 2. LightGBM Objesi (Her halükarda hazırla)
    lgb_model = LGBMClassifier(
        **lgb_params,
        random_state=42,
        n_jobs=-1
    )
    
    if architecture == 'Solo':
        best_model_name = recipe['architecture']['best_model']
        print(f"Mimari Karar: SOLO ({best_model_name}). Diğer model saf dışı bırakıldı.")
        final_model = xgb_model if best_model_name == 'XGBoost' else lgb_model
        
    elif architecture == 'Voting':
        print("Mimari Karar: HİBRİT (Soft Voting). Modeller güçlerini birleştiriyor.")
        final_model = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
            voting='soft'
        )
        
    print("\nModel Dev Set ile SIFIRDAN Eğitiliyor (Bu işlem biraz sürebilir)...")
    final_model.fit(X_train, y_train)
    print("Eğitim Tamamlandı! Validation işlemi bilerek es geçildi (Zaten optimum noktadayız).")
    
    return final_model

def seal_and_save_model(model, output_dir, model_name='final_cyber_model.pkl'):
    """
    Maksimum veriyle doyurulmuş nihai kapalı kutu sistemi diske kaydeder (Mühürleme).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model_name)
    
    joblib.dump(model, output_path)
    print(f"\n--- MÜHÜRLEME TAMAMLANDI ---")
    print(f"Nihai Model (Kapalı Kutu) şu konuma kaydedildi: {output_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dosya Yolları (Bu sefer sampled değil, saf HAM veriler çekiliyor!)
    train_pure_path = os.path.join(base_dir, "data", "processed", "X_train_ham.csv")
    val_pure_path = os.path.join(base_dir, "data", "processed", "X_val_ham.csv")
    features_path = os.path.join(base_dir, "configs", "selected_features.json")
    recipe_path = os.path.join(base_dir, "configs", "params.json")
    models_dir = os.path.join(base_dir, "models")
    
    print("=== 05_final_train.py (Nihai Dev Eğitim ve Mühürleme) Başlıyor ===")
    
    # Adım 1: Saf verileri birleştir (%85 Dev Set)
    X_dev, y_dev = load_and_combine_pure_data(train_pure_path, val_pure_path, features_path)
    
    # Adım 2: Dev sete Nihai SMOTE uygula
    X_dev_smote, y_dev_smote = apply_final_smote(X_dev, y_dev)
    
    # Adım 3: Port optimizasyonunu yap
    X_dev_smote = optimize_dtypes_for_trees(X_dev_smote)
    
    # Adım 4: Yepyeni bir model objesi yaratıp SADECE eğit (Fit)
    final_trained_model = build_and_train_final_model(X_dev_smote, y_dev_smote, recipe_path)
    
    # Adım 5: Eğitilmiş modeli diske kaydet (Mühürle)
    seal_and_save_model(final_trained_model, models_dir)
    
    print("=== Model Başarıyla Mühürlendi ===")

if __name__ == "__main__":
    main()