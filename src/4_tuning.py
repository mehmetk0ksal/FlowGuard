import pandas as pd
import numpy as np
import os
import json
import warnings

from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, recall_score, make_scorer

warnings.filterwarnings("ignore")

def load_data_and_features(train_path, val_path, features_path, label_column='Label'):
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

def calculate_targeted_smote_strategy(y):
    class_counts = y.value_counts()
    majority_count = class_counts.max()
    
    strategy = {}
    print("\n--- DoS Destekli SMOTE Stratejisi ---")
    
    for cls, count in class_counts.items():
        if count == majority_count:
            strategy[cls] = count
        elif cls == 3:  # DoS
            healthy_target = int(min(count * 20, majority_count * 0.20))
            strategy[cls] = max(count, healthy_target)
        else:
            healthy_target = int(min(count * 10, majority_count * 0.12))
            strategy[cls] = max(count, healthy_target)
        
        print(f"Sınıf {cls}: {count} -> {strategy[cls]}")
            
    return strategy

def apply_temporary_smote(X_train, y_train):
    print("\n--- BorderlineSMOTE Uygulanıyor ---")
    strategy = calculate_targeted_smote_strategy(y_train)
    
    smote = BorderlineSMOTE(sampling_strategy=strategy, random_state=42, kind='borderline-1')
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("\nSMOTE Sonrası Eğitim Seti Sınıf Dağılımı:")
    print(y_train_smote.value_counts().to_dict())
    
    return X_train_smote, y_train_smote

def optimize_dtypes_for_trees(X):
    X_opt = X.copy()
    categorical_candidates = ['Dst Port', 'Src Port']
    
    for col in categorical_candidates:
        if col in X_opt.columns:
            X_opt[col] = X_opt[col].astype('category')
            
    return X_opt

def create_sample_weights(y):
    """
    DoS'a daha yüksek ağırlık ver.
    """
    weight_map = {
        0: 1.0,   # Normal
        1: 2.0,   # Recon
        2: 2.5,   # Exploits
        3: 6.0,   # DoS
        4: 2.0    # Generic
    }
    return np.array([weight_map.get(label, 1.0) for label in y])

def custom_macro_plus_dos_recall(y_true, y_pred):
    macro = f1_score(y_true, y_pred, average='macro')
    dos_recall = recall_score((y_true == 3).astype(int), (y_pred == 3).astype(int))
    return 0.60 * macro + 0.40 * dos_recall

def tune_hyperparameters(X_train, y_train):
    print("\n--- Hiperparametre Optimizasyonu Başlıyor (DoS Öncelikli) ---")
    
    scorer = make_scorer(custom_macro_plus_dos_recall, greater_is_better=True)

    sample_weights = create_sample_weights(y_train)

    xgb_base = XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        tree_method='hist',
        device='cuda',
        enable_categorical=True
    )
    
    xgb_param_grid = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.3, 0.5],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 3, 5]
    }

    print("\n[>>> XGBoost Eğitimi Başlıyor <<<]")
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_grid,
        n_iter=20,
        scoring=scorer,
        cv=3,
        random_state=42,
        n_jobs=1,
        verbose=2
    )
    xgb_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    lgb_base = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        objective='multiclass'
    )
    
    lgb_param_grid = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [4, 6, 8, 10, -1],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'min_child_samples': [10, 20, 30, 50],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0, 1, 3, 5]
    }

    print("\n[>>> LightGBM Eğitimi Başlıyor <<<]")
    lgb_search = RandomizedSearchCV(
        estimator=lgb_base,
        param_distributions=lgb_param_grid,
        n_iter=20,
        scoring=scorer,
        cv=3,
        random_state=42,
        n_jobs=1,
        verbose=2
    )
    lgb_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    return xgb_search.best_estimator_, xgb_search.best_params_, lgb_search.best_estimator_, lgb_search.best_params_

def evaluate_and_decide_architecture(xgb_model, lgb_model, X_val, y_val):
    print("\n--- Saf Validation Seti Üzerinde Test ve Mimari Kararı ---")
    
    y_pred_xgb = xgb_model.predict(X_val)
    y_pred_lgb = lgb_model.predict(X_val)
    
    xgb_macro = f1_score(y_val, y_pred_xgb, average='macro')
    lgb_macro = f1_score(y_val, y_pred_lgb, average='macro')

    xgb_dos = recall_score((y_val == 3).astype(int), (y_pred_xgb == 3).astype(int))
    lgb_dos = recall_score((y_val == 3).astype(int), (y_pred_lgb == 3).astype(int))

    xgb_score = 0.60 * xgb_macro + 0.40 * xgb_dos
    lgb_score = 0.60 * lgb_macro + 0.40 * lgb_dos
    
    print(f"\n=> XGBoost Macro F1: %{xgb_macro*100:.2f}")
    print(f"=> XGBoost DoS Recall: %{xgb_dos*100:.2f}")
    print(f"=> XGBoost Birleşik Skor: %{xgb_score*100:.2f}")

    print(f"\n=> LightGBM Macro F1: %{lgb_macro*100:.2f}")
    print(f"=> LightGBM DoS Recall: %{lgb_dos*100:.2f}")
    print(f"=> LightGBM Birleşik Skor: %{lgb_score*100:.2f}")
    
    score_diff = abs(xgb_score - lgb_score)
    architecture_decision = {}
    
    if score_diff > 0.015:
        winner = "XGBoost" if xgb_score > lgb_score else "LightGBM"
        print(f"\nKarar (Solo): {winner} önde.")
        architecture_decision['strategy'] = 'Solo'
        architecture_decision['best_model'] = winner
    else:
        print("\nKarar (Hibrit - Soft Voting): İki model yakın. Voting seçildi.")
        architecture_decision['strategy'] = 'Voting'
        architecture_decision['models'] = ['XGBoost', 'LightGBM']
        
    return architecture_decision

def save_recipe(architecture_decision, xgb_params, lgb_params, output_dir):
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
    
    print("=== 04_tuning.py (DoS Öncelikli Optimizasyon) Başlıyor ===")
    
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