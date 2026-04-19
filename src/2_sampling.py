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

def print_distribution(df, label_column='Label', title="Dağılım"):
    print(f"\n--- {title} ---")
    counts = df[label_column].value_counts().sort_index()
    total = len(df)
    class_names = {
        0: "Normal",
        1: "Recon",
        2: "Exploits",
        3: "DoS",
        4: "Generic"
    }
    for cls, count in counts.items():
        pct = (count / total) * 100
        print(f"Sınıf {cls} ({class_names.get(cls, cls):<8}) -> {count} adet | %{pct:.4f}")

def perform_targeted_sampling(df, label_column='Label'):
    """
    1) Normal trafiği azaltır
    2) Saldırı sınıflarını kendi içinde daha dengeli hale getirir
    3) Özellikle DoS'u eğitimde daha görünür yapar
    """
    print_distribution(df, label_column, title="Orijinal Eğitim Verisi")

    normal_df = df[df[label_column] == 0]
    attack_df = df[df[label_column] != 0]

    class_counts = attack_df[label_column].value_counts().sort_index().to_dict()

    # Saldırı sınıfları içindeki en büyük sınıfı referans al
    max_attack_count = max(class_counts.values())

    # Hedef saldırı sınıf büyüklükleri
    # DoS'a özel boost veriyoruz
    target_attack_counts = {}
    for cls, count in class_counts.items():
        if cls == 3:  # DoS
            target_attack_counts[cls] = min(max_attack_count, count * 4)
        else:
            target_attack_counts[cls] = min(max_attack_count, count * 2)

    print("\n--- Hedef Saldırı Sınıf Boyutları ---")
    for cls, target in target_attack_counts.items():
        print(f"Sınıf {cls} hedef -> {target}")

    sampled_attack_parts = []

    for cls, group in attack_df.groupby(label_column):
        target_n = target_attack_counts[cls]

        if len(group) >= target_n:
            sampled_group = group.sample(n=target_n, random_state=42)
        else:
            # Oversampling with replacement
            deficit = target_n - len(group)
            extra = group.sample(n=deficit, replace=True, random_state=42)
            sampled_group = pd.concat([group, extra], ignore_index=True)

        sampled_attack_parts.append(sampled_group)

    sampled_attacks = pd.concat(sampled_attack_parts, ignore_index=True)

    # Normal trafik: toplam saldırının 2 katı kadar bırak
    target_normal_count = len(sampled_attacks) * 2

    print(f"\nToplam örneklenmiş saldırı: {len(sampled_attacks)}")
    print(f"Hedef normal trafik: {target_normal_count}")

    if len(normal_df) > target_normal_count:
        sampled_normal = normal_df.sample(n=target_normal_count, random_state=42)
    else:
        sampled_normal = normal_df

    balanced_df = pd.concat([sampled_normal, sampled_attacks], ignore_index=True)
    balanced_df = shuffle(balanced_df, random_state=42).reset_index(drop=True)

    print_distribution(balanced_df, label_column, title="Sampling Sonrası Yeni Eğitim Verisi")
    return balanced_df

def save_sampled_data(df, output_path):
    """
    Dengelenmiş yeni eğitim setini diske kaydeder.
    """
    df.to_csv(output_path, index=False)
    print(f"\nBaşarı: Dengelenmiş eğitim seti kaydedildi -> {output_path}")
    print(f"Dengelenmiş veri setinin toplam satır sayısı: {len(df)}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "processed", "X_train_ham.csv")
    output_file = os.path.join(base_dir, "data", "processed", "X_train_sampled.csv")
    
    print("=== 02_sampling.py (Hedefli Dengeleme) Başlıyor ===")
    
    train_df = load_training_data(input_file)
    balanced_train_df = perform_targeted_sampling(train_df)
    save_sampled_data(balanced_train_df, output_file)
    
    print("=== Dengeleme Başarıyla Tamamlandı ===")

if __name__ == "__main__":
    main()