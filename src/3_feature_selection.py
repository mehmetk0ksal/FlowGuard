import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier

def load_sampled_data(file_path):
    """
    Bir önceki adımda dengelenmiş olan eğitim verisini yükler.
    """
    print(f"Dengelenmiş veri yükleniyor: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hata: {file_path} bulunamadı! Lütfen dizini ve önceki adımları kontrol edin.")
    
    return pd.read_csv(file_path)

def drop_highly_correlated_features(X, threshold=0.90):
    print(f"\n--- Korelasyon Filtresi (Eşik: %{threshold*100}) ---")
    
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    
    # Hangi sütunun kime benzediğini bulup ekrana yazdıran kısım
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

def calculate_feature_importances(X, y):
    """
    Random Forest algoritması kullanarak veri setindeki her bir sütunun 
    hedefi (Label) tahmin etmedeki matematiksel önemini hesaplar.
    """
    print("Ağaç tabanlı algoritma (Random Forest) ile özellik önemi hesaplanıyor...")
    
    # n_jobs=-1 ile tüm işlemci çekirdekleri kullanılarak hızlandırma sağlanır
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    
    importances = rf_model.feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # En önemliden en önemsize doğru sırala
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return importance_df

def select_features_hybrid(importance_df, min_features=15, max_features=25, threshold=0.95):
    """
    Kümülatif önem (%95) hedefini, Min(15) ve Max(25) sınırları ile birleştiren hibrit yöntem.
    """
    # Kümülatif toplamı hesapla
    importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
    
    # %95'i (0.95) yakalamak için gereken sütun sayısını bul
    threshold_reached_indices = importance_df[importance_df['Cumulative_Importance'] >= threshold].index
    
    if len(threshold_reached_indices) > 0:
        target_count = threshold_reached_indices[0] + 1
    else:
        target_count = len(importance_df)
    
    # Hibrit Sınır Kontrolleri (Min/Max)
    if target_count < min_features:
        final_count = min_features
        reason = f"Kümülatif %{threshold*100} hedefine {target_count} sütunda ulaşıldı, ancak Minimum Sınır ({min_features}) uygulandı."
    elif target_count > max_features:
        final_count = max_features
        reason = f"Kümülatif %{threshold*100} hedefine {target_count} sütunda ulaşıldı, ancak Maksimum Sınır ({max_features}) uygulandı."
    else:
        final_count = target_count
        reason = f"Kümülatif %{threshold*100} hedefine {target_count} sütunda tam olarak ulaşıldı (Min-Max aralığına uygun)."

    # Karar verilen sayı kadar özelliği filtrele
    selected_features_df = importance_df.head(final_count)
    selected_features_list = selected_features_df['Feature'].tolist()
    
    print(f"\n--- Hibrit Özellik Seçimi Sonuçları ---")
    print(f"Ağaca Giren Toplam Özellik Sayısı: {len(importance_df)}")
    print(f"Karar Mekanizması: {reason}")
    print(f"Seçilen Nihai Özellik Sayısı: {final_count}\n")
    print("Seçilen Özellikler ve Katkıları:")
    
    for index, row in selected_features_df.iterrows():
        print(f"{index + 1}. {row['Feature']:<25} -> Bireysel: %{row['Importance']*100:.2f} | Kümülatif: %{row['Cumulative_Importance']*100:.2f}")
        
    return selected_features_list

def save_selected_features(features_list, output_dir, file_name='selected_features.json'):
    """
    Seçilen özelliklerin listesini JSON formatında kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(features_list, f, ensure_ascii=False, indent=4)
        
    print(f"\nSeçilen özelliklerin listesi kaydedildi: {output_path}")

def main():
    # Dinamik ve güvenli dizin tanımlamaları
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "processed", "X_train_sampled.csv")
    output_directory = os.path.join(base_dir, "configs")
    label_column = 'Label'
    
    print("=== 03_feature_selection.py (Hibrit Özellik Seçimi) Başlıyor ===")
    
    # Adım 1: Veriyi yükle ve X, y olarak ayır
    df = load_sampled_data(input_file)
    X = df.drop(columns=[label_column])
    y = df[label_column]
    
    # YENİ ADIM: Yüksek Korelasyonlu (Kopya) Sütunları Temizle
    # Eşik değerini %90 olarak belirledik, dilerseniz 0.95 yapabilirsiniz.
    X_reduced = drop_highly_correlated_features(X, threshold=0.90)
    
    # Adım 2: Feature Importance hesapla (Artık temizlenmiş X_reduced kullanılıyor)
    importance_df = calculate_feature_importances(X_reduced, y)
    
    # Adım 3: Hibrit yöntemle sütunları seç (Min 15, Max 25, Kümülatif %95)
    selected_features = select_features_hybrid(importance_df, min_features=15, max_features=25, threshold=0.95)
    
    # Adım 4: Listeyi kaydet
    save_selected_features(selected_features, output_directory)
    
    print("=== Özellik Seçimi Başarıyla Tamamlandı ===")

if __name__ == "__main__":
    main()