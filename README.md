---
title: 3E Vaka Calisma
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
---

# 3E Elektro Optik — AI Destekli Görsel Kalite Kontrol Sistemi (PoC)

[![Canlı Demo](https://img.shields.io/badge/HuggingFace-Spaces_Canlı_Demo-blue)](https://huggingface.co/spaces/ugurhandasdemir/3e-vaka-calisma)
[![GitHub Kod](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Ugurhandasdemir/3e-vaka-calisma)
[![Teknik Doküman](https://img.shields.io/badge/Doküman-teknik--dokuman.pdf-red)](docs/teknik-dokuman.pdf)

Bu proje, **3E Elektro Optik** elektro-optik ve elektro-mekanik ürün gruplarının (optik lens grupları, termal kamera modülleri, elektro-optik gözetleme sistemleri) üretim ve montaj hatlarında kullanılmak üzere geliştirilmiş yapay zeka destekli bir **görsel kalite kontrol (Visual Quality Control - QC) prototipidir (PoC)**.

Sistem; muayene uzmanının yerini almak yerine operatörü hızlandırmak, insan yorgunluğundan kaynaklanan gözden kaçırmaları engellemek ve AS9100 standartlarında denetlenebilir bir kalite kontrol kaydı oluşturmak üzere **insan-döngüde (human-in-the-loop)** hibrit karar destek mekanizması sunar.

Canlı demo Hugging Face Spaces üzerinde **ZeroGPU donanımında CPU çıkarımı (CPU inference)** ile çalışmaktadır.

---

## 🏭 İki Aşamalı Üretim ve Kalite Kapısı Mimarisi

Sistem, parça seri numaralarını ve denetim kayıtlarını ortak paylaşan iki temel üretim aşaması ve birleşik kalite kapısından oluşur:

1. **🏭 1. Aşama — Üretim İçi Görsel Muayene (Visual AI):**
   - **Görsel AI Mimarisi:** visual AI: anomaly PatchCore-WRN50 (+DINOv2 ensemble planned), YOLO11n, Gemini
   - Montaj öncesi ve montaj esnasında lens, sensör ve gövde yüzey kusurları denetlenir; uzman operatör MobileSAM etkileşimli segmentasyonu ile etiketleme yaparak aktif öğrenme döngüsünü besler.
2. **🧪 2. Aşama — Üretim Sonrası Test (Post-Production Optical Measurement):**
   - **Optik Ölçüm:** post-production: OpenCV sub-pixel reticle measurement, max error 0.06 px on synthetic targets
   - Montajı tamamlanan elektro-optik sistemlerin kolimatör retikülü üzerinde Huber M-tahmincisiyle alt-piksel optik eksen (boresight) kaçıklığı (mrad), görünür-termal kanal hizalaması ve titreşim öncesi/sonrası mekanik eksen kayması (drift) ölçülür.
3. **🚦 Kalite Kapısı (Quality Gate):**
   - Parça seri numarası bazında izlenir: Yalnızca görsel muayene insan kararı **KABUL** (ACCEPT) ve son test **GEÇTİ** (PASS) olduğunda (ve titreşim kayması tolerans içindeyse) **SEVKE HAZIR** (🟢) statüsü verilir; aşamalardan biri eksikse **BEKLEMEDE** (🟡), herhangi bir aşama başarısızsa **RET** (🔴) verilir.

---

## ⚡ Değerlendirme için Hızlı Deneme

Sistemi 1 dakikada canlı olarak test etmek için:
1. [Hugging Face Spaces Canlı Demosunu](https://huggingface.co/spaces/ugurhandasdemir/3e-vaka-calisma) açın.
2. Örnek galerisinden hazır bir parça görseli seçin (ör. *Optik lens grubu* veya *Termal kamera modülü*).
3. **"Parçayı Analiz Et"** butonuna basın.
4. Çıkan yapay zeka önerisini ve motor analizlerini inceleyip nihai kararınızı (Kabul / Ret / İnceleme) seçin.
5. **"Kararı Kaydet"** butonuna basarak muayeneyi veritabanına işleyin.
6. **"Geçmiş ve Pano"** sekmesine geçerek kaydedilen denetim izini ve istatistikleri görüntüleyin.

---

## 🚀 Canlı Demo Akışı (5 Dakikalık Sunum Senaryosu)

1. **Sağlam Numune Analizi (KABUL):**
   - Sol panelden *Optik lens grubu* seçilir ve örnek galerisinden sağlam bir lens görseli yüklenir.
   - **"Parçayı Analiz Et"** butonuna basılır.
   - Anomali motoru $p$-değerini yüksek ($p > 0.50$) hesaplar, YOLO dedektörü kusur bulamaz.
   - **Erken çıkış (Early exit)** tetiklenerek VLM çağrılmadan **KABUL ÖNERİSİ** üretilir.
2. **Kusurlu Numune Analizi (RET):**
   - Galeriden çizikli veya montaj kusurlu bir görsel seçilir.
   - Anomali motoru belirgin ısı haritası üretir; YOLO dedektörü kusuru sınırlar ve sınıflandırır; Gemini VLM kusurun türünü ve neden kabul edilemeyeceğini Türkçe muhakeme ile raporlar.
   - Sistem **RET ÖNERİSİ** üretir, sağ panelde anomali ısı haritası ve tespit kutusu yan yana görüntülenir.
3. **Sınır Durum ve Yüksek Risk Koruması (İNSAN İNCELEMESİ - REVIEW):**
   - Motorlar arasında çelişki olan veya geçmişte geri çağırma riski bulunan yüksek riskli parça grubunda (*Termal kamera modülü*) bir numune analiz edilir.
   - Yüksek riskli termal grup **asla erken çıkış almaz**, tüm motorlar zorunlu koşturulur.
   - Sistem **İNSAN İNCELEMESİ** önerir. Arayüz doğrudan "AI önerisini onayla" seçeneğini kapatarak operatörü fiziksel inceleme sonucuna göre doğrudan **Kabul** veya **Ret** seçmeye yönlendirir.
4. **Kayıt ve Aktif Öğrenme (Geçmiş ve Pano):**
   - Operatör nihai kararını veritabanına kaydeder.
   - **"Geçmiş ve Pano"** sekmesine geçilerek AI-insan uyum oranı, insan incelemesine düşme yüzdesi ve kusur dağılımı incelenir.
   - **"YOLO Etiket Paketini İndir"** butonuyla aktif öğrenme döngüsünü besleyecek veri seti dışa aktarılır.

---

## 🏗️ Sistem Mimarisi

```text
[ Muayene Görseli (JPG/PNG) ]
           │
           ├─────────────────────────────────────────┐
           ▼                                         ▼
┌─────────────────────────┐               ┌─────────────────────────┐
│     Anomali Motoru      │               │     Kusur Dedektörü     │
│ EfficientAD-S (ONNX)    │               │     YOLO11n (ONNX)      │
│  (Yedek: PatchCore)     │               │   (ultralytics inference)│
└──────────┬──────────────┘               └──────────┬──────────────┘
           │ [Isı haritası + p-değeri]               │ [Kutular + Sınıf + Güven]
           └────────────────┬────────────────────────┘
                            │
                   ┌────────┴────────┐
                   │   Erken Çıkış   │──(p > 0.50 & Dedektör boş & Termal Değil)──► [ KABUL ÖNERİSİ ]
                   │   Kontrolü      │
                   └────────┬────────┘
                            │ (Şüphe / kusur var veya Yüksek Riskli Termal Grup)
                            ▼
                 ┌─────────────────────────┐
                 │    VLM Akıl Yürütme     │
                 │  gemini-3.5-flash-lite  │ (Orijinal + Isı Haritası + Kutular)
                 │ (Yedek: 3.8-fl, 3.1-fl) │
                 └──────────┬──────────────┘
                            │ [Kusur Tipi + Muhakeme + Şiddet]
                            ▼
                 ┌─────────────────────────┐
                 │     Hibrit Füzyon       │ Conformal p + Ağırlıklı Oylama
                 │  & Belirsizlik Kapısı   │ Anomali Olasılığı = clip(log p / log 0.02)
                 └──────────┬──────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
[ KABUL ÖNERİSİ ]   [ İNSAN İNCELEMESİ ]   [ RET ÖNERİSİ ]
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
                 ┌─────────────────────────┐
                 │ Operatör Nihai Kararı   │ (Kabul / Ret / İnceleme)
                 └──────────┬──────────────┘
                            │
                            ▼
                 ┌─────────────────────────┐
                 │     SQLite Veritabanı   │ (Kayıt, Metrikler, AS9100 İzlenebilirlik)
                 └──────────┬──────────────┘
                            │
                            ├──────────────────────────┐
                            ▼                          ▼
                 [ ERP/MES CSV Dışa Aktarım ]   [ YOLO ZIP Aktif Öğrenme ]
```

---

## 💻 Teknoloji Yığını ve Model Detayları

- **Arayüz:** Gradio 6 (Blocks API, Soft Theme, mobil uyumlu)
- **Anomali Tespiti (Öneri Katmanı):**
  - **PatchCore:** `wide_resnet50_2` omurgası ile referans hafıza bankası yedeği (CPU ~120–150 ms).
  - **EfficientAD-S:** `notebooks/01_efficientad.ipynb` ile eğitilmiş ONNX modelleri (`metal_nut` Image AUROC 0.9868). `models/anomaly_<kategori>.onnx` mevcut olduğunda **otomatik devreye girer**.
- **Kusur Dedektörü (Tespit Katmanı):**
  - **YOLO11n DETECTION:** `notebooks/02_yolo11n.ipynb` ile 10 MVTec AD sınıfından gerçek kusurlarla ve çevrimdışı veri artırımıyla (~1.000 eğitim görseli) eğitilmiş ONNX nesne tespit modeli.
  - **Metrikler:** Genel test kümesinde **mAP50: 0.814** (mAP50-95: 0.518), uygulamada kullanılan 3 hedef kategoride **mAP50: 0.779** (`scratch`: 0.813, `coating`: 0.839, `connector_assembly`: 0.790).
- **Görsel Muhakeme (VLM Katmanı):**
  - **Google Gemini 3.5 Flash-Lite** (Birincil model). Kota veya hata durumunda otomatik devreye giren yedek modeller: **gemini-3.8-flash**, **gemini-3.1-flash-lite**.
  - Pydantic yapılandırılmış JSON çıktısı, sıcaklık 0.4, 3 paralel sorgu ile çoğunluk oylaması.
- **Karar Füzyonu:**
  - Conformal prediction kalibrasyonu: Anomali olasılığı $a = \text{clip}(\log p / \log 0.02, 0, 1)$ olarak logaritmik ölçeklenir.
  - Ağırlıklar: Anomali: 0.45, Dedektör: 0.25, VLM: 0.30.
  - **Yüksek Risk Kuralı:** Geri çağırma geçmişi olan yüksek riskli ürün grubu (*Termal kamera modülü*) **asla erken çıkış almaz**, tüm modeller çalıştırılır ve herhangi bir kusur oyunda otomatik kabul engellenir.
- **Temsili Ürün Grupları (MVTec AD Eşleşmesi):**
  - Optik lens grubu $\rightarrow$ `metal_nut`
  - Termal kamera modülü $\rightarrow$ `transistor` (Yüksek Risk)
  - Gözetleme ünitesi $\rightarrow$ `cable`
- **Operatör Etiketleme ve SAM Desteği:** Operatör arayüzünde etkileşimli segmentasyon ve kutu etiketleme için Segment Anything (SAM / MobileSAM) entegrasyonu **geliştiriliyor** (aktif geliştirme aşamasındadır).
- **Veri Tabanı ve Dışa Aktarma:** SQLite (`data/qc.db`), CSV denetim izi ve YOLO formatında aktif öğrenme ZIP paketi.

---

## 📁 Proje Yapısı

```text
3e-vaka-calisma/
├── app.py                     # Gradio kullanıcı arayüzü ve olay yönetimi
├── config.py                  # Model eşikleri, kategoriler ve sistem sabitleri
├── pipeline.py                # Muayene iş akışı (Anomali -> Dedektör -> Erken Çıkış / VLM -> Füzyon)
├── fusion.py                  # Karar füzyon kuralları ve conformal anomali olasılık hesabı
├── calib.py                   # Conformal p-değeri kalibrasyon modülü
├── storage.py                 # SQLite veritabanı kayıt ve aktif öğrenme ZIP dışa aktarımı
├── engines/
│   ├── anomaly.py             # EfficientAD-S (ONNX) ve PatchCore (wide_resnet50_2) motoru
│   ├── detector.py            # YOLO11n nesne tespit motoru (ultralytics / ONNX)
│   └── vlm.py                 # Gemini VLM istemcisi (3.5 Flash-Lite + yedekler)
├── models/
│   ├── yolo.onnx              # Eğitilmiş YOLO11n tespit modeli (ONNX)
│   ├── yolo_metrics.json      # YOLO11n doğrulama ve test metrikleri (mAP50: 0.814)
│   └── calib_*.npy            # Kategori bazlı kalibrasyon skorları
├── notebooks/
│   ├── 01_efficientad.ipynb   # EfficientAD-S eğitim ve ONNX dışa aktarım not defteri
│   └── 02_yolo11n.ipynb       # YOLO11n tespit eğitimi ve çevrimdışı artırım not defteri
├── docs/
│   ├── teknik-dokuman.md      # Ayrıntılı teknik mimari ve doğrulama raporu
│   ├── teknik-dokuman.pdf     # Yazdırılabilir ve dağıtılabilir teknik doküman
│   └── CONTRACT.md            # Modüller arası veri yapıları ve sözleşmeler
└── samples/                   # Kategori bazlı sağlam ve kusurlu muayene test görselleri
```

---

## 🔧 Yerel Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın ve Sanal Ortamı Hazırlayın
```bash
git clone https://github.com/Ugurhandasdemir/3e-vaka-calisma.git
cd 3e-vaka-calisma

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ortam Değişkenlerini Tanımlayın
`.env.example` dosyasını `.env` olarak kopyalayın ve Gemini API anahtarınızı girin:
```bash
cp .env.example .env
# .env dosyasını düzenleyin:
# GEMINI_API_KEY=AIzaSy...
# GEMINI_MODEL=gemini-3.5-flash-lite
```

### 3. Uygulamayı Başlatın
```bash
python app.py
```
Uygulama yerel ağınızda `http://0.0.0.0:7860` adresinde sunulacaktır.

---

## 📂 `models/` Klasörü ve Model Eğitimi

Sistem açılışta `models/` klasöründeki modelleri arar. Modeller mevcut olmadığında sistem **zarif bir şekilde (graceful fallback)** alternatiflere geçer:

- `models/anomaly_<kategori>.onnx`: `notebooks/01_efficientad.ipynb` ile eğitilen EfficientAD-S modelleri. Varsa otomatik olarak kullanılır, aksi takdirde PatchCore (`wide_resnet50_2`) hafıza bankası yedeği devrededir.
- `models/calib_<kategori>.npy`: Conformal prediction kalibrasyon skor dizileri.
- `models/yolo.onnx`: `notebooks/02_yolo11n.ipynb` ile eğitilen YOLO11n ONNX tespit modeli. (Yoksa dedektör devre dışı kalır ve ağırlık diğer motorlara paylaştırılır).

---

## 🔄 Aktif Öğrenme (Active Learning) Döngüsü

Kalite kontrol sistemlerinin üretim hattında zamanla olgunlaşması için aktif öğrenme döngüsü entegre edilmiştir:

1. **Geri Bildirim Toplama:** Operatör tarafından **RET** kararı verilen ve tespit kutusu içeren numuneler SQLite veritabanında işaretlenir.
2. **Paketleme:** Pano sekmesindeki **"YOLO Etiket Paketini İndir (ZIP)"** butonu, bu hatalı numunelerin görsellerini, `classes.txt` sınıf fihristini ve normalize edilmiş YOLO koordinat etiketlerini ZIP arşivi olarak sunar.
3. **Yeniden Eğitim:** İndirilen ZIP paketi, Colab notebook'una (`notebooks/02_yolo11n.ipynb`) veya yerel eğitim hattına beslenerek dedektörün zayıf olduğu köşe durumlar (edge-cases) hızla modele kazandırılır.

---

## 🔒 Güvenlik ve Uyumluluk Notları

- **API Anahtarı Güvenliği:** Google Gemini API anahtarı repoda asla tutulmaz; yerel `.env` veya HF Spaces Secrets üzerinden yüklenir.
- **Girdi Doğrulama:** Yüklenen görseller bellek taşmalarına ve zararlı dosyalara karşı doğrulanır (Maks. 10 MB, RGBA/CMYK formatından RGB'ye dönüştürme, en uzun kenarı 1024 piksele ölçekleme).
- **EXIF Temizliği:** Yüklenen görseller bellekte PIL ile yeniden kodlanarak cihaz/konum üstverileri (EXIF) temizlenir.
- **Kuyruk ve Hız Sınırı:** Gradio kuyruğu (`concurrency_limit=2`) ile API kotası ve sunucu kaynakları korunur.
- **Savunma Sanayii / On-Prem Tavsiyesi:** PoC sürümünde VLM çağrıları bulut API üzerinden yapılmaktadır. Askeri ve savunma sanayii standartlarında (AS9100) üretim hattı için kapalı devre (on-prem) yerel VLM (Qwen2.5-VL vb.) ve yerel GPU kümesi kullanılmalıdır.

---

## ⚠️ Sınırlılıklar

- **Temsili Veri Seti:** Bu çalışma bir kavram kanıtlama (PoC) olup gerçek elektro-optik parça verileri yerine MVTec Anomaly Detection veri kümesi (`metal_nut`, `transistor`, `cable`) kullanılmıştır.
- **Fiziksel Ölçüm Sınırı:** Optik eksen kaçıklığı ve konektör tork gevşekliği gibi kusurlar salt 2D görüntüden %100 doğrulanamaz; optik kolimatör ve tork ölçer sensör verisi entegrasyonu gerektirir.
- **Geçici Disk:** Hugging Face Spaces ücretsiz katmanında disk kalıcı değildir; konteyner yeniden başladığında SQLite veritabanı sıfırlanır. Demo tutarlılığı için açılışta temsili geçmiş veriler (`seed_demo`) yüklenir.

---

## 📄 Veri Lisansı ve Atıf

Bu projede kullanılan temsili endüstriyel numuneler **MVTec AD** veri kümesinden derlenmiştir:
- **Kaynak:** MVTec Software GmbH, *MVTec Anomaly Detection Dataset (MVTec AD)*, Bergmann et al., CVPR 2019.
- **Lisans:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- Yalnızca araştırma, eğitim ve kavram kanıtlama (PoC) amacıyla temsili parça olarak kullanılmıştır.

