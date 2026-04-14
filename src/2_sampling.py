import pandas as pd
import os
from sklearn.utils import shuffle

def load_training_data(file_path):
    """
    Sadece eğitim verisini (X_train_ham) belleğe yükler.
    Validation ve Test setlerine kesinlikle dokunulmaz.
    """
    print(f"Eğitim verisi yükleniyor: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hata: {file_path} bulunamadı! Önce 01_preprocessing.py çalıştırılmalı.")
    
    return pd.read_csv(file_path)

def perform_smart_undersampling(df, label_column='Label'):
    """
    Saldırıların tamamını korur.
    Normal trafikten, toplam saldırı sayısının tam 3 katı kadar rastgele örneklem çeker.
    Böylece %25 Saldırı, %75 Normal trafik oranı yakalanır.
    """
    # 1. Saldırılar (Label != 0) ve Normal Trafik (Label == 0) olarak veriyi ikiye böl
    attacks = df[df[label_column] != 0]
    normal_traffic = df[df[label_column] == 0]
    
    total_attacks_count = len(attacks)
    target_normal_count = total_attacks_count * 3
    
    print("--- Örnekleme (Sampling) İstatistikleri ---")
    print(f"Mevcut Toplam Saldırı Adedi: {total_attacks_count}")
    print(f"Mevcut Normal Trafik Adedi: {len(normal_traffic)}")
    print(f"Hedeflenen Normal Trafik Adedi (3 Katı): {target_normal_count}")
    
    # 2. Undersampling (Alt Örnekleme) İşlemi
    if len(normal_traffic) < target_normal_count:
        print("Uyarı: Veri setindeki normal trafik, hedeflenen 3 katı sayıdan daha az!")
        print("Mevcut olan tüm normal trafik alınıyor...")
        sampled_normal = normal_traffic
    else:
        # Normal trafikten rastgele örneklem çek (random_state=42 ile tekrarlanabilirlik sağlanır)
        sampled_normal = normal_traffic.sample(n=target_normal_count, random_state=42)
    
    # 3. Altın Veri Setini Birleştirme
    balanced_df = pd.concat([attacks, sampled_normal])
    
    # 4. Veriyi Karıştırma (Modelin sırayı ezberlemesini önlemek için çok kritiktir)
    balanced_df = shuffle(balanced_df, random_state=42).reset_index(drop=True)
    
    print("--- Birleştirme Tamamlandı ---")
    print(f"Yeni Veri Seti Dağılımı:")
    print(balanced_df[label_column].value_counts(normalize=True) * 100) # Yüzdelik oranları göster
    
    return balanced_df

def save_sampled_data(df, output_path):
    """
    Dengelenmiş yeni eğitim setini diske kaydeder.
    """
    df.to_csv(output_path, index=False)
    print(f"\nBaşarı: Dengelenmiş eğitim seti kaydedildi -> {output_path}")
    print(f"Dengelenmiş veri setinin toplam satır sayısı: {len(df)}")

def main():
    # Dosya yolları
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "processed", "X_train_ham.csv")
    output_file = os.path.join(base_dir, "data", "processed", "X_train_sampled.csv")
    
    print("=== 02_sampling.py (Dengeleme) Başlıyor ===")
    
    # Adım 1: Sadece saf eğitim kasasını yükle
    train_df = load_training_data(input_file)
    
    # Adım 2: Akıllı Undersampling işlemini uygula
    balanced_train_df = perform_smart_undersampling(train_df)
    
    # Adım 3: Sonucu kaydet
    save_sampled_data(balanced_train_df, output_file)
    
    print("=== Dengeleme Başarıyla Tamamlandı ===")

if __name__ == "__main__":
    main()