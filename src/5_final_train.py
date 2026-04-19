import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings

from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore")

def load_and_combine_pure_data(train_path, val_path, features_path, label_column='Label'):
    print(f"Saf Train Seti yükleniyor: {train_path}")
    print(f"Saf Validation Seti yükleniyor: {val_path}")
    
    df_train_pure = pd.read_csv(train_path)
    df_val_pure = pd.read_csv(val_path)
    
    df_dev = pd.concat([df_train_pure, df_val_pure], ignore_index=True)
    
    with open(features_path, 'r', encoding='utf-8') as f:
        selected_features = json.load(f)
        
    X_dev = df_dev[selected_features]
    y_dev = df_dev[label_column]
    
    print(f"\nDevasa Saf Eğitim Seti (Dev Set) oluşturuldu. Toplam Satır: {len(df_dev)}")
    return X_dev, y_dev

def calculate_targeted_smote_strategy(y):
    class_counts = y.value_counts()
    majority_count = class_counts.max()
    strategy = {}
    
    for cls, count in class_counts.items():
        if count == majority_count:
            strategy[cls] = count
        elif cls == 3:  # DoS
            strategy[cls] = int(min(count * 20, majority_count * 0.20))
            strategy[cls] = max(strategy[cls], count)
        else:
            strategy[cls] = int(min(count * 10, majority_count * 0.12))
            strategy[cls] = max(strategy[cls], count)
            
    return strategy

def apply_final_smote(X_dev, y_dev):
    print("\n--- Nihai BorderlineSMOTE Uygulanıyor (Dev Set Üzerine) ---")
    print("Orijinal Dev Set Sınıf Dağılımı:")
    print(y_dev.value_counts().to_dict())
    
    strategy = calculate_targeted_smote_strategy(y_dev)
    
    smote = BorderlineSMOTE(sampling_strategy=strategy, random_state=42, kind='borderline-1')
    X_dev_smote, y_dev_smote = smote.fit_resample(X_dev, y_dev)
    
    print("\nNihai SMOTE Sonrası Dev Set Sınıf Dağılımı:")
    print(y_dev_smote.value_counts().to_dict())
    
    return X_dev_smote, y_dev_smote

def optimize_dtypes_for_trees(X):
    X_opt = X.copy()
    categorical_candidates = ['Dst Port', 'Src Port']
    for col in categorical_candidates:
        if col in X_opt.columns:
            X_opt[col] = X_opt[col].astype('category')
    return X_opt

def create_sample_weights(y):
    weight_map = {
        0: 1.0,
        1: 2.0,
        2: 2.5,
        3: 6.0,
        4: 2.0
    }
    return np.array([weight_map.get(label, 1.0) for label in y])

def build_and_train_final_model(X_train, y_train, recipe_path):
    print(f"\n--- Nihai Model İnşası ve Eğitim Başlıyor ---")
    print(f"Tarifin Sırrı (Parametreler) Okunuyor: {recipe_path}")
    
    with open(recipe_path, 'r', encoding='utf-8') as f:
        recipe = json.load(f)
        
    architecture = recipe['architecture']['strategy']
    xgb_params = recipe['parameters']['XGBoost']
    lgb_params = recipe['parameters']['LightGBM']

    sample_weights = create_sample_weights(y_train)
    
    xgb_model = XGBClassifier(
        **xgb_params,
        random_state=42,
        eval_metric='mlogloss',
        tree_method='hist',
        device='cuda',
        enable_categorical=True
    )
    
    lgb_model = LGBMClassifier(
        **lgb_params,
        random_state=42,
        n_jobs=-1
    )
    
    if architecture == 'Solo':
        best_model_name = recipe['architecture']['best_model']
        print(f"Mimari Karar: SOLO ({best_model_name})")
        final_model = xgb_model if best_model_name == 'XGBoost' else lgb_model
        
        print("\nModel Dev Set ile eğitiliyor...")
        final_model.fit(X_train, y_train, sample_weight=sample_weights)
        
    elif architecture == 'Voting':
        print("Mimari Karar: HİBRİT (Soft Voting)")
        
        print("\nXGBoost ayrı eğitiliyor...")
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

        print("\nLightGBM ayrı eğitiliyor...")
        lgb_model.fit(X_train, y_train, sample_weight=sample_weights)

        final_model = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
            voting='soft'
        )
        
        # VotingClassifier fitted estimator beklediği için fit çağrısı gerekir
        # ama burada base estimators zaten eğitildiğinden yeniden fit ediyoruz
        # basitlik için aynı veride tekrar fit edelim
        print("\nVoting modeli ortak eğitimden geçiriliyor...")
        final_model.fit(X_train, y_train)

    print("Eğitim Tamamlandı!")
    return final_model

def seal_and_save_model(model, output_dir, model_name='final_cyber_model.pkl'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model_name)
    
    joblib.dump(model, output_path)
    print(f"\n--- MÜHÜRLEME TAMAMLANDI ---")
    print(f"Nihai Model şu konuma kaydedildi: {output_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    train_pure_path = os.path.join(base_dir, "data", "processed", "X_train_ham.csv")
    val_pure_path = os.path.join(base_dir, "data", "processed", "X_val_ham.csv")
    features_path = os.path.join(base_dir, "configs", "selected_features.json")
    recipe_path = os.path.join(base_dir, "configs", "params.json")
    models_dir = os.path.join(base_dir, "models")
    
    print("=== 05_final_train.py (DoS Güçlendirmeli Final Eğitim) Başlıyor ===")
    
    X_dev, y_dev = load_and_combine_pure_data(train_pure_path, val_pure_path, features_path)
    X_dev_smote, y_dev_smote = apply_final_smote(X_dev, y_dev)
    X_dev_smote = optimize_dtypes_for_trees(X_dev_smote)
    
    final_trained_model = build_and_train_final_model(X_dev_smote, y_dev_smote, recipe_path)
    seal_and_save_model(final_trained_model, models_dir)
    
    print("=== Model Başarıyla Mühürlendi ===")

if __name__ == "__main__":
    main()