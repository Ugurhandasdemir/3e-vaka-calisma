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

Bu proje, **3E Elektro Optik** elektro-optik ve elektro-mekanik ürün gruplarının (optik lens grupları, termal kamera modülleri, elektro-optik gözetleme sistemleri) üretim ve montaj hatlarında kullanılmak üzere geliştirilmiş yapay zeka destekli bir **görsel kalite kontrol (Visual Quality Control - QC) prototipidir (PoC)**.

Sistem, muayene uzmanının yerini almak yerine; operatörü hızlandırmak, insan yorgunluğundan kaynaklanan gözden kaçırmaları engellemek ve standart, denetlenebilir bir kalite kontrol kaydı oluşturmak üzere **insan-döngüde (human-in-the-loop)** hibrit karar destek mekanizması sunar.

---

## 🚀 Canlı Demo Akışı (5 Dakikalık Sunum Senaryosu)

1. **Sağlam Numune Analizi (KABUL):**
   - Sol panelden *Optik lens grubu* seçilir ve örnek galerisinden sağlam bir lens görseli yüklenir.
   - **"Parçayı Analiz Et"** butonuna basılır.
   - Anomali motoru $p$-değerini yüksek ($p > 0.50$) hesaplar, YOLO dedektörü kusur bulamaz.
   - **Erken çıkış (Early exit)** tetiklenerek VLM çağrılmadan **KABUL ÖNERİSİ** (%95+ güven) üretilir.
2. **Kusurlu Numune Analizi (RET):**
   - Galeriden çizikli veya bozuk konektörlü bir görsel seçilir.
   - Anomali motoru belirgin ısı haritası üretir; YOLO dedektörü kusuru sınırlar ve sınıflandırır; Gemini VLM kusurun türünü ve neden kabul edilemeyeceğini Türkçe muhakeme ile raporlar.
   - Sistem **RET ÖNERİSİ** üretir, sağ panelde anomali ısı haritası ve tespit kutusu yan yana görüntülenir.
3. **Sınır Durum ve Belirsizlik (İNSAN İNCELEMESİ - REVIEW):**
   - Motorlar arasında çelişki olan veya geri çağırma riski yüksek parça grubunda (*Termal kamera modülü*) sınır bir numune analiz edilir.
   - Sistem **İNSAN İNCELEMESİ** önerir. Arayüz "Onayla" seçeneğini kapatarak operatörü fiziksel inceleme sonucuna göre doğrudan **Kabul** veya **Ret** seçmeye zorlar.
4. **Kayıt ve Aktif Öğrenme (Geçmiş ve Pano):**
   - Operatör nihai kararını veritabanına kaydeder.
   - **"Geçmiş ve Pano"** sekmesine geçilerek AI-insan uyum oranı, insan incelemesine düşme yüzdesi ve kusur dağılımı incelenir.
   - **"YOLO Etiket Paketini İndir"** butonuna basılarak aktif öğrenme döngüsünü besleyecek veri seti dışa aktarılır.

---

## 🏗️ Sistem Mimarisi

```text
[ Muayene Görseli (JPG/PNG) ]
           │
           ├─────────────────────────────────────────┐
           ▼                                         ▼
┌─────────────────────────┐               ┌─────────────────────────┐
│     Anomali Motoru      │               │     Kusur Dedektörü     │
│ EfficientAD (ONNX)      │               │   YOLO11s-seg (ONNX)    │
│  (Yedek: PatchCore)     │               │   (ultralytics inference)│
└──────────┬──────────────┘               └──────────┬──────────────┘
           │ [Isı haritası + p-değeri]               │ [Kutular + Sınıf + Güven]
           └────────────────┬────────────────────────┘
                            │
                   ┌────────┴────────┐
                   │   Erken Çıkış   │──(Anomali yok & YOLO boş)──► [ KABUL ÖNERİSİ ]
                   │   Kontrolü      │
                   └────────┬────────┘
                            │ (Şüphe veya kusur var)
                            ▼
                 ┌─────────────────────────┐
                 │    VLM Akıl Yürütme     │
                 │ Google Gemini 3.5 Flash-Lite │ (Orijinal + Isı Haritası + Kutular)
                 └──────────┬──────────────┘
                            │ [Kusur Tipi + Muhakeme + Şiddet]
                            ▼
                 ┌─────────────────────────┐
                 │     Hibrit Füzyon       │ Conformal p + Ağırlıklı Oylama
                 │  & Belirsizlik Kapısı   │ Güven Skoru + Risk Kuralı Kontrolü
                 └──────────┬──────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
[ KABUL ÖNERİSİ ]   [ İNSAN İNCELEMESİ ]   [ RET ÖNERİSİ ]
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
                 ┌─────────────────────────┐
                 │ Operatör Nihai Kararı   │ (Onay / Müdahale / Düzeltme)
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

## 💻 Teknoloji Yığını (Tech Stack)

- **Arayüz:** Gradio 6 (Blocks API, Soft Theme, mobil uyumlu)
- **Anomali Tespiti:** EfficientAD-S (ONNX Runtime) & PatchCore (PyTorch ResNet-18 hafıza bankası yedeği)
- **Nesne Tespiti / Segmentasyon:** YOLO11s-seg ONNX
- **Görsel Muhakeme (VLM):** Google Gemini 3.5 Flash-Lite (Yapılandırılmış JSON şeması, sıcaklık 0.4)
- **Füzyon:** Conformal Prediction $p$-değeri, eşik oylaması ve belirsizlik kapısı
- **Veri Tabanı & Depolama:** SQLite (`data/qc.db`), PNG görsel deposu
- **Dışa Aktarma:** Standart CSV ve YOLOv8/11 formatında etiketlenmiş aktif öğrenme ZIP paketi

---

## 🔧 Yerel Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın ve Sanal Ortamı Hazırlayın
```bash
git clone https://github.com/Ugurhandasdemir/3e-qc-poc.git
cd 3e-qc-poc

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
*(Alternatif olarak terminalde `export GEMINI_API_KEY="anahtarınız"` komutunu çalıştırabilirsiniz).*

### 3. Uygulamayı Başlatın
```bash
python app.py
```
Uygulama yerel ağınızda `http://0.0.0.0:7860` adresinde sunulacaktır.

---

## 📂 `models/` Klasörü ve Model Eğitimi

Sistem açılışta `models/` klasöründeki modelleri arar. Modeller mevcut olmadığında sistem **zarif bir şekilde (graceful fallback)** yedek motorlara geçer:

- `models/anomaly_<kategori>.onnx`: Colab `notebooks/01_efficientad.ipynb` ile eğitilen EfficientAD modelleri. (Yoksa PatchCore referans hafıza bankası kullanılır).
- `models/anomaly_<kategori>.json`: Model girdi/çıktı adları, normalizasyon ve eşik değerleri meta verisi.
- `models/calib_<kategori>.npy`: Conformal prediction $p$-değeri kalibrasyon skor dizisi.
- `models/yolo.onnx`: Colab `notebooks/02_yolo_seg.ipynb` ile fine-tune edilmiş YOLO11s segmentasyon/tespit modeli. (Yoksa dedektör devre dışı kalır ve ağırlık diğer motorlara paylaştırılır).

---

## 🔄 Aktif Öğrenme (Active Learning) Döngüsü

Kalite kontrol sistemlerinin üretim hattında zamanla olgunlaşması için aktif öğrenme döngüsü entegre edilmiştir:

1. **Geri Bildirim Toplama:** Operatör tarafından **RET** kararı verilen ve tespit kutusu içeren numuneler SQLite veritabanında işaretlenir.
2. **Paketleme:** Pano sekmesindeki **"YOLO Etiket Paketini İndir (ZIP)"** butonu, bu hatalı numunelerin görsellerini, `classes.txt` sınıf fihristini ve normalize edilmiş YOLO koordinat etiketlerini ZIP arşivi olarak sunar.
3. **Yeniden Eğitim:** İndirilen ZIP paketi, Colab notebook'una veya yerel eğitim hattına beslenerek dedektörün zayıf olduğu köşe durumlar (edge-cases) hızla modele kazandırılır.

---

## 🔒 Güvenlik ve Uyumluluk Notları

- **API Anahtarı Güvenliği:** Google Gemini API anahtarı repoda asla tutulmaz; yerel `.env` veya HF Spaces Secrets üzerinden yüklenir.
- **Girdi Doğrulama:** Yüklenen görseller bellek taşmalarına ve zararlı dosyalara karşı doğrulanır (Maks. 10 MB, RGBA/CMYK formatından RGB'ye dönüştürme, en uzun kenarı 1024 piksele ölçekleme).
- **EXIF Temizliği:** Yüklenen görseller bellekte PIL ile yeniden kodlanarak cihaz/konum üstverileri (EXIF) temizlenir.
- **Kuyruk ve Hız Sınırı:** Gradio kuyruğu (`concurrency_limit=2`) ile API kotası ve sunucu kaynakları korunur.
- **Savunma Sanayii / On-Prem Tavsiyesi:** PoC sürümünde VLM çağrıları bulut API üzerinden yapılmaktadır. Askeri ve savunma sanayii standartlarında (AS9100) üretim hattı için kapalı devre (on-prem) yerel VLM (Qwen2.5-VL vb.) ve yerel GPU kümesi kullanılmalıdır.

---

## ⚠️ Sınırlılıklar

- **Temsili Veri Seti:** Bu çalışma bir kavram kanıtlama (PoC) olup gerçek elektro-optik parça verileri yerine MVTec Anomaly Detection veri kümesi kullanılmıştır.
- **Fiziksel Ölçüm Sınırı:** Optik eksen kaçıklığı ve konektör tork gevşekliği gibi kusurlar salt 2D görüntüden %100 doğrulanamaz; optik kolimatör ve tork ölçer sensör verisi entegrasyonu gerektirir.
- **Geçici Disk:** Hugging Face Spaces ücretsiz katmanında disk kalıcı değildir; konteyner yeniden başladığında SQLite veritabanı sıfırlanır. Demo tutarlılığı için açılışta temsili geçmiş veriler (`seed_demo`) yüklenir.

---

## 📄 Veri Lisansı ve Atıf

Bu projede kullanılan temsili endüstriyel numuneler **MVTec AD** veri kümesinden derlenmiştir:
- **Kaynak:** MVTec Software GmbH, *MVTec Anomaly Detection Dataset (MVTec AD)*, Bergmann et al., CVPR 2019.
- **Lisans:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- Yalnızca araştırma, eğitim ve kavram kanıtlama (PoC) amacıyla temsili parça olarak kullanılmıştır.
