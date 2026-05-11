# Data Science Project 29 — California Housing: Fiyat Tahmini

**Modül**: ML-02 (Regresyon) • **Süre**: 2-3 saat

## 🎯 Proje Senaryosu

Bir emlak danışmanlık şirketinde **junior data scientist** olarak işe başladın. Müşterilere ev fiyatı tahmini sunan bir online platform geliştiriyorlar. Senin görevin: California ev fiyatlarını feature'lardan (gelir, oda sayısı, konum...) tahmin eden bir **regresyon modeli** kurmak.

Ama dikkat — sadece "model çalışıyor" yetmez. Yöneticin sana şöyle bir uyarı verdi:

> *"Sayılara güveniyoruz, ama hangi feature'ların gerçekten önemli olduğunu da bilmek istiyoruz. Katsayıların stabil olduğundan emin ol — özellikle multicollinearity meselesini kontrol et."*

Bu projede ML-02 dersinde öğrendiklerini birleştirip uygulayacaksın:

- ✅ **Çoklu lineer regresyon** (8 feature, gerçek veri)
- ✅ **train_test_split** + **StandardScaler** (lineer regresyonda invariance var ama yorum için şart)
- ✅ **MAE, RMSE, R²** metrikleri ile değerlendirme
- ✅ **Katsayı yorumlama** — ölçekli model üzerinden feature önemi
- ✅ **Korelasyon analizi** ile multicollinearity tespiti
- ✅ **VIF (Variance Inflation Factor)** — istatistiksel ölçüm (manuel hesap)
- ✅ **Feature engineering** — anlamlı oran üretme (BedrmsPerRoom)
- ✅ **Yeni veri tahmini** + tam pipeline

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-29

# Virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# veya: venv\Scripts\activate     # Windows

# Bağımlılıklar
pip install -r requirements.txt
```

## 📝 Yapacakların

`tasks/task_manager.py` içinde **10 fonksiyon**:

| # | Fonksiyon | Ne Yapar |
|---|-----------|----------|
| 1 | `load_california_data()` | sklearn'den California Housing yükle |
| 2 | `split_and_scale()` | train/test böl + StandardScaler uygula |
| 3 | `train_linear_model()` | LinearRegression eğit |
| 4 | `evaluate_model()` | MAE, RMSE, R² hesapla |
| 5 | `analyze_coefficients()` | Katsayıları feature'larla eşle, sırala |
| 6 | `find_correlated_pairs()` | |korelasyon| > eşik olan çiftler |
| 7 | `calculate_vif()` | Her feature için VIF (manuel formül) |
| 8 | `engineer_bedrms_per_room()` | Korelasyonlu feature'ları birleştir |
| 9 | `predict_house_price()` | Yeni ev için fiyat tahmini |
| 10 | `run_full_pipeline()` | Hepsini bir araya getir |

Her fonksiyonun `pass` kısmını docstring'e göre doldur.

## ✅ Testleri Çalıştır

**Otomatik (önerilen)**: dosya değiştikçe testleri çalıştırır
```bash
python watch.py
```

**Manuel**:
```bash
pytest tests/test_question.py -v
```

Toplam **19 test**. Hepsi PASS olunca projen tamam.

## 💡 İpuçları ve Beklenen Sayılar

### MAE / RMSE / R²
- Lineer regresyon ile California verisinde **R² ≈ 0.58** civarı (base model)
- **MAE ≈ 0.53** (yani ortalama ±53.000 dolar yanılma)
- **RMSE ≈ 0.75** (RMSE her zaman MAE ≥)

Eğer R²'n çok düşükse, scaling'i atlama veya yanlış sıra olabilir.

### Multicollinearity ipuçları
- California'da iki şüpheli çift var:
  - `AveRooms` ↔ `AveBedrms` (korelasyon ≈ 0.84)
  - `Latitude` ↔ `Longitude` (korelasyon ≈ −0.92)
- VIF eşikleri: **> 5** yumuşak şüphe, **> 10** sert problem

### VIF Formülü Hatırlatma
```
VIF_i = 1 / (1 - R²_i)
```
Burada `R²_i` = "i'nci feature'ı diğer feature'lardan tahmin eden yardımcı regresyonun R²'si".
- R² = 0 → VIF = 1 (temiz)
- R² = 0.80 → VIF = 5 (yumuşak)
- R² = 0.90 → VIF = 10 (sert)

### Feature Engineering Sezgisi
`AveBedrms` ve `AveRooms` ayrı ayrı yerine ikisini **anlamlı bir orana** çevir:
```
BedrmsPerRoom = AveBedrms / AveRooms
```
Bu hem korelasyonu kırar hem yorum gücü kazanır ("oda başına yatak odası oranı").

## 🎓 Pedagojik Hedefler

Bu projeyi bitirdiğinde:
- Çoklu lineer regresyonun matematiğini ve sınırlarını biliyorsun
- Multicollinearity'yi **gözle (heatmap), kanıtla (bootstrap), ölçüyle (VIF)** üç farklı yoldan tanıyorsun
- Pratik çözümleri (sil, birleştir, Ridge köprüsü) uygulayabiliyorsun
- "Modelim çalışıyor" ile "katsayılarım güvenilir" arasındaki **mühendislik farkını** anlıyorsun

## 🚀 Sonraki Adımlar

Bu projeyi bitirdiğinde **DS-30 (Polinom + Ridge/Lasso)** projesine geç. Orada bu modeli **regularization ile geliştireceksin**.

İyi çalışmalar 🏠📊
