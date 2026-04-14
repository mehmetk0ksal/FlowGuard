import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    """
    Belirtilen yoldan veriyi yükler ve tekrar eden ağ paketlerini (kopyaları) temizler.
    """
    print(f"Veri yükleniyor: {file_path}")
    df = pd.read_csv(file_path)
    
    initial_row_count = len(df)
    
    # Kopya temizliği (Deduplication)
    df.drop_duplicates(inplace=True)
    
    final_row_count = len(df)
    print(f"Kopya temizliği tamamlandı. Silinen satır: {initial_row_count - final_row_count}")
    print(f"Eşsiz satır sayısı: {final_row_count}")
    
    return df

def handle_missing_and_infinite_values(df):
    """
    Veri setindeki sonsuz (inf) değerleri tespit eder, NaN değerine çevirir 
    ve ardından NaN içeren tüm satırları veri setinden temizler.
    """
    initial_row_count = len(df)
    
    # Sonsuz değerleri (inf ve -inf) NaN'a çevir
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # NaN değer barındıran satırları sil
    df.dropna(inplace=True)
    
    final_row_count = len(df)
    print(f"Inf/NaN temizliği tamamlandı. Silinen bozuk satır sayısı: {initial_row_count - final_row_count}")
    print(f"Son durumdaki satır sayısı: {final_row_count}")
    
    return df

def prevent_data_leakage(df):
    """
    Modelin 'kimin yaptığını' ezberlemesini önlemek için tanımlayıcı sütunları siler.
    """
    columns_to_drop = ['Src IP', 'Dst IP', 'Timestamp', 'Flow ID']
    
    # Sadece veri setinde var olan sütunları silmek için kontrol
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns:
        df.drop(columns=existing_columns, inplace=True)
        print(f"Veri sızıntısını önlemek için silinen sütunlar: {existing_columns}")
    else:
        print("Veri sızıntısını önlemek için belirtilen sütunlar veri setinde bulunamadı.")
    
    return df

def map_categories(df, label_column='Label'):
    """
    Dağınık alt saldırı tiplerini 5 ana gruba manuel bir sözlük ile dönüştürür ve sayısallaştırır.
    """
    # Alt kategorileri ana kategorilere dönüştüren sözlük yapısı
    category_dict = {
        'Benign': 0,
        'Reconnaissance': 1, 
        'Analysis': 1,
        'Exploits': 2, 
        'Shellcode': 2, 
        'Backdoor': 2, 
        'Worms': 2,
        'DoS': 3,
        'Fuzzers': 4, 
        'Generic': 4
    }

    if label_column in df.columns:
        df[label_column] = df[label_column].map(category_dict)
        # Eşleşmeyen ve NaN (Boş) dönen satırlar varsa düşürülür
        df.dropna(subset=[label_column], inplace=True)
        df[label_column] = df[label_column].astype(int)
        print("Kategori haritalama ve sayısallaştırma işlemi başarıyla tamamlandı.")
    else:
        print(f"Uyarı: '{label_column}' sütunu veri setinde bulunamadı!")
        
    return df

def split_and_save_data(df, output_dir, label_column='Label'):
    """
    Veriyi Train (%70), Validation (%15) ve Test (%15) olarak tabakalı şekilde böler.
    Saf kasalar olarak belirlenen hedefe kaydeder.
    """
    X = df.drop(columns=[label_column])
    y = df[label_column]
    
    # Veriyi önce %70 Train ve %30 (Validation + Test) olarak böl
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    # Kalan %30'luk veriyi kendi içinde ikiye böl (%15 Validation, %15 Test)
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    # Train, Val ve Test setlerini kendi etiketleriyle birleştirip kaydedelim
    train_df = pd.concat([X_train_raw, y_train_raw], axis=1)
    val_df = pd.concat([X_val_raw, y_val_raw], axis=1)
    test_df = pd.concat([X_test_raw, y_test_raw], axis=1)
    
    # Kayıt dizinini kontrol et ve yoksa oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, 'X_train_ham.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'X_val_ham.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'X_test_ham.csv'), index=False)
    
    print(f"Saf veriler şu dizine kaydedildi: {output_dir}")
    print(f"Train Seti: {len(train_df)} satır")
    print(f"Validation Seti: {len(val_df)} satır")
    print(f"Test Seti: {len(test_df)} satır")

def main():
    # Dosya yolları
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "raw", "CICFlowMeter.csv")
    output_directory = os.path.join(base_dir, "data", "processed")
    
    print("--- Ön İşleme (Preprocessing) Başlıyor ---")
    
    # Adım 1: Veri Yükleme ve Kopya Temizliği
    df = load_and_clean_data(input_file)
    
    # EK ADIM: Inf ve NaN değerlerin temizlenmesi
    df = handle_missing_and_infinite_values(df)
    
    # Adım 2: Veri Sızıntısını Önleme (Sabit liste ile)
    df = prevent_data_leakage(df)
    
    # Adım 3: Üst Kategori Haritalama ve Sayısallaştırma
    df = map_categories(df)
    
    # Adım 5: Orijinal Veriyi Bölme ve Koruma
    split_and_save_data(df, output_directory)
    
    print("--- Ön İşleme Başarıyla Tamamlandı ---")

if __name__ == "__main__":
    main()