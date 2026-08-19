# Perakende Talep Tahmini ve Stok Optimizasyonu

XGBoost tabanlı bir talep tahmin modeli geliştirdim ve bu tahmini, veriye dayalı (dağılım varsayımsız) bir güvenlik stoğu/reorder point stratejisine çevirdim. Amaç: perakende zincirlerinde stok fazlası ve stok yokluğu maliyetlerini azaltacak, uçtan uca bir karar destek sistemi kurmak.

## Problem

Perakende mağazalarında talep tahmini, stok planlamasının temelidir. Kötü tahmin iki maliyete yol açar: **stok fazlası** (raf üzerinde atıl sermaye) ve **stok yokluğu** (kayıp satış). Bu proje, mağaza-gün bazında satış talebini tahmin edip, bu tahmini gerçek bir envanter kararına (ne zaman, ne kadar sipariş verilmeli) dönüştürüyor.

## Veri

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) — 1.115 mağaza, ~2.5 yıllık günlük satış verisi (2013-01 – 2015-07), mağaza meta verisiyle (tip, assortment, promosyon programı, rakip mesafesi) zenginleştirilmiş.

## Yöntem

**1. EDA:** Pazar kapanışları, promosyon etkisi (~%39 satış artışı), 2014 ortasında ~165 mağazanın geçici kapanması, ve veri kalitesi sorunları (StateHoliday'de str/int karışıklığı, mağaza kapanışlarında takvim süreksizliği) tespit edildi ve düzeltildi.

**2. Feature Engineering:** Tarih özellikleri, lag (7/14/28 gün), rolling ortalama/std (7/14/30 gün), promosyon/tatil flag'leri, rakip açılış durumu. Takvim süreksizliği (mağaza kapanışları) nedeniyle oluşan hatalı lag/rolling değerleri tespit edilip NaN'a çevrildi.

**3. Modelleme:** Naive baseline (geçen haftanın aynı günü) → XGBoost. Zaman bazlı train/test split (son 6 hafta test), 3 katlamalı expanding-window time series CV ile sonucun sağlamlığı doğrulandı.

**4. Model Validasyonu:** SHAP ile feature importance analizi — Promo ve geçmiş satış (lag/rolling) özellikleri en baskın sinyaller. Mağaza tipine göre segmentlenmiş modeller de denendi; global model ile karşılaştırıldığında ek fayda sağlamadığı görüldü (segmentasyon veri hacmini bölerek modelleri zayıflattı).

**5. Stok Optimizasyonu:** Tahmin hatasından, dağılım varsayımsız (ampirik persentil tabanlı) güvenlik stoğu hesaplandı. Aykırı mağazaların etkisini azaltmak için MAD (Median Absolute Deviation) kullanıldı, ve mağazalar arası "risk havuzlama" ile istatistiksel örneklem büyütüldü. Kalibrasyon, veri sızıntısını önlemek için ayrı bir out-of-sample dönemden yapıldı.

## Sonuçlar

| Metrik | Naive Baseline | XGBoost |
|---|---|---|
| RMSE | 2809 | 888 |
| MAPE | %31.82 | %9.12 |
| CV MAPE (3 katlama ort.) | - | %8.54 (±0.2) |

**İş çıktısı:** Aynı hizmet seviyesini (%95 stoksuz kalma garantisi) korurken, model tabanlı yaklaşım naive'e kıyasla gereken güvenlik stoğu yatırımını **%72** azaltıyor — bu oran %90-%99 arası tüm hizmet seviyelerinde tutarlı (%70-75 aralığında).

## Sınırlamalar ve Öğrenilenler

- Basit bir $ maliyet simülasyonu denendi, ancak 6 haftalık test döneminde birkaç öngörülemeyen, açıklanamayan talep sıçraması (örn. tek bir mağazada normalin 3-4 katı satış) sonucu domine etti ve güvenilir bir karşılaştırma sağlamadı. Bunun yerine hizmet-seviyesi-eşleştirmeli envanter verimliliği metriği tercih edildi.
- Mağazaları geçmiş volatiliteye göre risk segmentlerine ayırmak denendi; bu, geçmişte hiç iz bırakmamış ani olayları yakalayamadı — kısa bir gözlem penceresinin doğal bir sınırı, daha uzun geçmiş veri gerektirir.
- Tedarik süresi (5 gün) ve maliyet oranları (elde tutma %0.3, stoksuzluk %30) gerçek şirket verisi olmadığı için makul varsayımlarla belirlendi.

## Nasıl çalıştırılır

```bash
pip install -r requirements.txt
```
Notebook'ları sırayla çalıştır: `01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_modeling.ipynb`

## Kullanılan araçlar

Python, Pandas, XGBoost, SHAP, Scikit-learn, Matplotlib/Seaborn