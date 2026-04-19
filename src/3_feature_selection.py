import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier

def load_sampled_data(file_path):
    print(f"Dengelenmiş veri yükleniyor: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hata: {file_path} bulunamadı! Lütfen dizini ve önceki adımları kontrol edin.")
    return pd.read_csv(file_path)

def drop_highly_correlated_features(X, threshold=0.90):
    print(f"\n--- Korelasyon Filtresi (Eşik: %{threshold*100}) ---")
    
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    
    for column in upper_triangle.columns:
        correlated_with = upper_triangle.index[upper_triangle[column] > threshold].tolist()
        if correlated_with:
            for match in correlated_with:
                benzerlik = upper_triangle.loc[match, column] * 100
                print(f"Kopya Tespit: '{column}' -> '{match}' ile %{benzerlik:.1f} aynı.")
            to_drop.add(column)
            
    to_drop = list(to_drop)
    
    if len(to_drop) > 0:
        print(f"\nToplam {len(to_drop)} sütun siliniyor...")
        X_reduced = X.drop(columns=to_drop)
    else:
        print("Yüksek korelasyonlu sütun bulunamadı.")
        X_reduced = X
        
    print(f"Kalan benzersiz özellik sayısı: {len(X_reduced.columns)}\n")
    return X_reduced

def calculate_feature_importances_multiclass(X, y):
    print("Multiclass Random Forest ile global feature importance hesaplanıyor...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    rf_model.fit(X, y)

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return importance_df

def calculate_feature_importances_dos_vs_rest(X, y):
    print("DoS-vs-Rest Random Forest ile DoS odaklı importance hesaplanıyor...")
    y_binary = (y == 3).astype(int)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf_model.fit(X, y_binary)

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance_DoS': rf_model.feature_importances_
    }).sort_values(by='Importance_DoS', ascending=False).reset_index(drop=True)

    return importance_df

def select_hybrid_features(global_importance_df, dos_importance_df,
                           base_feature_count=30, dos_feature_count=12, max_total=40):
    """
    Final feature set:
    - Global top feature'lar
    - DoS için özel önemli feature'lar
    """
    global_features = global_importance_df.head(base_feature_count)['Feature'].tolist()
    dos_features = dos_importance_df.head(dos_feature_count)['Feature'].tolist()

    final_features = []
    for feat in global_features + dos_features:
        if feat not in final_features:
            final_features.append(feat)

    final_features = final_features[:max_total]

    print("\n--- Hibrit Feature Seçimi Sonuçları ---")
    print(f"Global Feature Sayısı: {len(global_features)}")
    print(f"DoS-Özel Feature Sayısı: {len(dos_features)}")
    print(f"Nihai Toplam Feature Sayısı: {len(final_features)}")
    print("\nSeçilen Nihai Feature Listesi:")
    for i, feat in enumerate(final_features, 1):
        print(f"{i}. {feat}")

    return final_features

def save_selected_features(features_list, output_dir, file_name='selected_features.json'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(features_list, f, ensure_ascii=False, indent=4)
        
    print(f"\nSeçilen özelliklerin listesi kaydedildi: {output_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "processed", "X_train_sampled.csv")
    output_directory = os.path.join(base_dir, "configs")
    label_column = 'Label'
    
    print("=== 03_feature_selection.py (DoS Destekli Hibrit Seçim) Başlıyor ===")
    
    df = load_sampled_data(input_file)
    X = df.drop(columns=[label_column])
    y = df[label_column]
    
    X_reduced = drop_highly_correlated_features(X, threshold=0.90)

    global_importance_df = calculate_feature_importances_multiclass(X_reduced, y)
    dos_importance_df = calculate_feature_importances_dos_vs_rest(X_reduced, y)

    selected_features = select_hybrid_features(
        global_importance_df,
        dos_importance_df,
        base_feature_count=30,
        dos_feature_count=12,
        max_total=40
    )
    
    save_selected_features(selected_features, output_directory)
    
    print("=== Özellik Seçimi Başarıyla Tamamlandı ===")

if __name__ == "__main__":
    main()