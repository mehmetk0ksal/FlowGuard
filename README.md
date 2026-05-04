# FlowGuard

# Demo Veri Setleri ve Dosya Yapısı

Projenin canlı demosunu (Streamlit arayüzü) test edebilmeniz için klasörde **6 adet** demo ve veri üretim dosyası bulunmaktadır:

*   **`real_` Ön Eki Taşıyan Dosyalar (`real_demo_traffic.csv` & `real_demo_traffic_50.csv`):** Orijinal CICFlowMeter test setinden (eğitimde kullanılmamış `X_test_ham` verisinden) cımbızla çekilmiş, %100 gerçek ağ trafiği kayıtlarıdır. Modelin gerçek dünya performansını görmek için kullanılır.
*   **Ön Eki Olmayan Dosyalar (`demo_traffic.csv` & `demo_traffic_50.csv`):** Gemini yardımıyla tamamen teorik ve abartılı saldırı (DoS, Exploits vb.) senaryolarıyla oluşturulmuş sentetik verilerdir. Modelin uç (extreme) değerlerde nasıl davrandığını ve ezber yapıp yapmadığını test etmek için tasarlanmıştır.
*   **Python Betikleri (`.py`):** Sentetik ve gerçek veri oluşturan bu Python kodları, güncel olarak **30 satırlık** demo dosyaları (sonunda 50 yazmayanlar) üretecek şekilde ayarlanmıştır. İsimlerinin sonunda `_50` olan CSV dosyaları ise aynı mantıkla üretilmiş 50 satırlık daha geniş test setleridir.

---

# 🚀 Kurulum Öncesi Önemli Notlar (Drive Gereksinimleri)

Projeyi bilgisayarınızda çalıştırmadan önce **Mehmet'in Drive klasöründen** temin etmeniz gereken bazı eksik bileşenler bulunmaktadır:

1.  **Eksik Klasörler (`data/` ve `models/`):** Boyut sınırları nedeniyle ham veri setleri ve eğitilmiş model dosyaları (`final_cyber_model.pkl`) GitHub'a yüklenmemiştir. Uygulamanın çalışması için bu iki klasörü Drive'dan indirip proje ana dizinine atmalısınız.
2.  **LLM Entegrasyonu (API Key):** Projedeki Siber Güvenlik Uzmanı (LLM) analiz modülü **Cerebras AI** (Llama 3 modeli) altyapısını kullanmaktadır. Sistemin çalışması için Drive'daki yapılandırma dosyasının içindeki `APIKEY` değerini kopyalayıp, projenizdeki `configs/api_config.json` dosyasının içine yapıştırmanız gerekmektedir.
