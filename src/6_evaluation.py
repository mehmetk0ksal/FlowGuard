import pandas as pd
import numpy as np
import os
import json
import joblib
import time
from sklearn.metrics import f1_score, confusion_matrix, classification_report

def load_test_data_and_model(test_path, features_path, model_path, label_column='Label'):
    """
    Kilitli kasadaki hiç DOKUNULMAMIŞ X_test_ham verisini, 
    kullanılan özellikleri ve eğitilmiş/mühürlenmiş modeli yükler.
    """
    print(f"Hiç görülmemiş Test Seti yükleniyor: {test_path}")
    df_test = pd.read_csv(test_path)
    
    with open(features_path, 'r', encoding='utf-8') as f:
        selected_features = json.load(f)
        
    X_test = df_test[selected_features]
    y_test = df_test[label_column]
    
    print(f"Mühürlü Model yükleniyor: {model_path}")
    model = joblib.load(model_path)
    
    return X_test, y_test, model

def optimize_dtypes_for_trees(X):
    """Eğitimdeki port optimizasyonunun aynısını test setine de uygular."""
    X_opt = X.copy()
    categorical_candidates = ['Dst Port', 'Src Port']
    for col in categorical_candidates:
        if col in X_opt.columns:
            X_opt[col] = X_opt[col].astype('category')
    return X_opt

def save_performance_report(report_text, output_dir):
    """Hazırlanan raporu logs klasörüne tarih damgasıyla kaydeder."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Dosya adını o anki tarih ve saatle oluştur (Örn: final_report_20231027_1430.txt)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"final_report_{timestamp}.txt"
    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
        
    print(f"\n[BİLGİ] Performans raporu (Karne) şuraya mühürlendi: {output_path}")

def evaluate_real_world_performance(model, X_test, y_test):
    """
    Gerçek dünya testini yapar, istenen tüm metrikleri ve 
    ekstra güvenlik analizlerini raporlar. Aynı zamanda rapor metnini döndürür.
    """
    report_lines = []
    
    def log(message):
        """Hem ekrana basar hem de rapora ekler."""
        print(message)
        report_lines.append(message)

    log("\n--- Gerçek Dünya Sınavı (Final Test) Başladı ---")
    
    # 1. HIZ TESTİ (Inference Time)
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_packet = (total_time / len(X_test)) * 1000 # Milisaniye cinsinden
    
    log(f"Hız Testi: Toplam {len(X_test)} paket {total_time:.2f} saniyede analiz edildi.")
    log(f"Ortalama Gecikme: Paket başına {time_per_packet:.4f} milisaniye (ms).")

    # 2. GENEL F1 SKORLARI
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted') # Normal F1 dediğimiz ağırlıklı skor
    
    log(f"\n=> Genel F1-Score (Ağırlıklı - Normal): %{weighted_f1*100:.2f}")
    log(f"=> Genel F1-Score (Macro - Zorlayıcı): %{macro_f1*100:.2f}")
    
    # 3. SINIF BAZLI BİLME ORANLARI (Yüzde kaç oranla bildiği - Recall)
    log("\n--- Kategori Bazlı İsabet Oranları (Recall) ---")
    
    # Sözlüğümüz
    class_names = {
        0: "Normal",
        1: "Recon",
        2: "Exploits",
        3: "DoS",
        4: "Generic"
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Matrisin her satırı o sınıfın GERÇEK toplamını, köşegenler ise DOĞRU bilinenleri verir
    for i in range(len(cm)):
        total_real = np.sum(cm[i, :])
        correctly_predicted = cm[i, i]
        
        # Eğer test setinde o sınıftan hiç yoksa 0'a bölme hatası almamak için kontrol
        if total_real > 0:
            accuracy_rate = (correctly_predicted / total_real) * 100
            class_name = class_names.get(i, f"Tür {i}")
            log(f"Tür {i} ({class_name:<8}) -> %{accuracy_rate:.2f} başarı ile tespit edildi. (Bildiği: {correctly_predicted} / Toplam: {total_real})")
        else:
            log(f"Tür {i} test setinde hiç bulunamadı.")

    # 4. GÜVENLİK AÇIĞI ANALİZİ (False Negative: Saldırıyı Normal sanma durumu)
    log("\n--- Kritik Güvenlik (Kaçak) Analizi ---")
    
    total_attacks = 0
    missed_attacks = 0
    
    for i in range(1, len(cm)): # 0 (Normal) hariç diğerlerine bak
        total_attacks += np.sum(cm[i, :])
        missed_attacks += cm[i, 0] # Gerçekte saldırı (i) olup, modelin 0 (Normal) dedikleri
        
    if total_attacks > 0:
        miss_rate = (missed_attacks / total_attacks) * 100
        log(f"Test Setindeki Toplam Gerçek Saldırı: {total_attacks}")
        log(f"Modelin Normal Sanıp İÇERİ SIZDIRDIĞI Saldırı: {missed_attacks} (Kaçak Oranı: %{miss_rate:.2f})")
        
        if miss_rate < 5.0:
            log("[DURUM: MÜKEMMEL] Model sistemi bir kalkan gibi koruyor!")
        elif miss_rate < 15.0:
            log("[DURUM: İYİ] Kabul edilebilir düzeyde kaçak var, ancak kurallar sıkılaştırılabilir.")
        else:
            log("[DURUM: RİSKLİ] Tehlikeli alarm körlüğü mevcut, modele daha fazla saldırı verisi gösterilmeli.")

    log("\nDetaylı Skorbord (Classification Report):")
    log(classification_report(y_test, y_pred, target_names=[class_names.get(i, str(i)) for i in range(len(cm))]))

    # Toparlanan raporu metin olarak döndür
    return "\n".join(report_lines)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dosya Yolları
    test_path = os.path.join(base_dir, "data", "processed", "X_test_ham.csv")
    features_path = os.path.join(base_dir, "configs", "selected_features.json")
    model_path = os.path.join(base_dir, "models", "final_cyber_model.pkl")
    logs_dir = os.path.join(base_dir, "logs") # Yeni logs klasörü yolu
    
    print("=== 06_evaluation.py (Final Sınavı ve Raporlama) Başlıyor ===")
    
    # Adım 1: Test verisini, özellikleri ve modeli yükle
    X_test, y_test, loaded_model = load_test_data_and_model(test_path, features_path, model_path)
    
    # Adım 2: Eğitimde yapılan Port dönüşümünü testte de yap
    X_test_opt = optimize_dtypes_for_trees(X_test)
    
    # Adım 3: Sınavı başlat ve sonuç metnini al
    final_report_text = evaluate_real_world_performance(loaded_model, X_test_opt, y_test)
    
    # Adım 4: Raporu logs klasörüne kaydet
    save_performance_report(final_report_text, logs_dir)
    
    print("=== Proje Başarıyla Tamamlandı ===")

if __name__ == "__main__":
    main()