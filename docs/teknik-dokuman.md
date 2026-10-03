# 3E Elektro Optik — AI Destekli Görsel Kalite Kontrol Sistemi (PoC)
## Teknik Vaka Dokümanı ve Mimari Rapor

### 1. Problem ve Çözüm Özeti
Günlük 85 adet kritik elektro-optik muayene hacmi, üretim hattında biriken 1.200 adet etiketsiz parça görseli ve ayda ortalama 3 adet kaçan kusurun (özellikle geri çağırma riski barındıran termal kamera modüllerinde) yarattığı kalite riski temel operasyonel darboğazı oluşturmaktadır. Bu vaka çalışmasında; etiketsiz veriyle başlayabilen anomali tespiti, nesne dedektörü ve görsel dil modelini (VLM) conformal istatistiksel kalibrasyonla birleştiren, operatörü sürecin merkezinde tutan (human-in-the-loop) hibrit bir kalite kontrol karar destek sistemi geliştirilmiştir. Çözüm; operatörün yerini almak yerine muayene süresini kısaltmayı, kaçan kusurları sıfırlamayı ve AS9100 uyumlu tam izlenebilirlik sağlamayı hedefler.

---

### 2. Kullanılan AI Modelleri ve Teknolojileri

Sistem hiyerarşik bir **"iki aşamalı sistem"** mantığıyla kurgulanmıştır: Birinci aşamada anomali motoru etiketsiz bölge önerisi (RPN rolü) üretir; ikinci aşamada ise YOLO ve VLM bu aday bölgeleri sınıflandırıp bağlamsal olarak gerekçelendirir.

| Katman | Seçilen Model / Yöntem | Neden Seçildi? | Alternatifler ve Kıyas |
| :--- | :--- | :--- | :--- |
| **Anomali Tespiti** *(Aşama 1: Öneri)* | **PatchCore** (`wide_resnet50_2` omurga) + **EfficientAD-S** (ONNX çalışma zamanı) | PatchCore sıfır eğitimle, yalnızca sağlam numunelerin bellek bankasıyla çalışır; soğuk başlangıç ve 1.200 etiketsiz görsel için idealdir. EfficientAD-S ise diskte ONNX modeli bulunduğunda otomatik devreye girer; CPU'da ~100–300 ms, GPU'da 2–3 ms ile F/P lideridir. | Dinomaly (~%99,6 AUROC ile literatür lideridir ancak ViT-B omurgası CPU için ağırdır); Zero-shot CLIP (~%90–92 AUROC, endüstriyel ince kusurlarda zayıf kalmaktadır). |
| **Kusur Dedektörü** *(Aşama 2: Sınıflandırma)* | **YOLO11n / YOLO11s** (Colab ortamında MVTec maskelerinden eğitilmiş ONNX) | Hızlı çıkarım (~90 ms CPU), tek komutla ONNX uyumluluğu ve düşük bellek tüketimi. Bilinen kusur tiplerini doğrudan sınırlar ve etiketler. | Faster/Cascade/Mask R-CNN (küçük kusurlarda başarılı olsa da CPU gecikmesi 1–3 sn'dir ve ONNX dışa aktarım zinciri kırılgandır). |
| **Görsel Dil Modeli (VLM)** *(Aşama 2: Muhakeme)* | **Google Gemini 2.5 Flash** (Pydantic ile yapılandırılmış JSON şema, $T=0.4$, 3 paralel çağrı) | Yapılandırılmış çıktı garantisi; orijinal görsel ile anomali ısı haritası bindirmesini ve YOLO kutu koordinatlarını birlikte işler. Myriad makalesindeki (arXiv:2310.19070) "uzman ipucu" fikrinin token eğitimi gerektirmeyen, prompting tabanlı uyarlamasıdır. 3 paralel çağrı ile çoğunluk ve tutarlılık ($c/3$) hesaplanır. | GPT-4o / Gemini Pro (yüksek maliyet ve API gecikmesi); On-prem Qwen-VL (sunucu GPU'su gerektirir; açık kaynak üretim hedefidir). |
| **Karar Füzyonu & Belirsizlik** | **Conformal Prediction $p$-değeri** + Ağırlıklı Oylama + 3 Bölge + Risk Kuralı | Siyah kutu skorlar yerine istatistiksel temelli belirsizlik yönetimi. Ağırlıklar: Anomali 0.45, Dedektör 0.25, VLM 0.30. Yüksek riskli grupta (*Termal kamera modülü*) tek bir kusur oyu dahi otomatik kabulü engeller; doğrudan insan incelemesine yönlendirir. | Saf lojistik regresyon veya kural tabansız LLM füzyonu (halüsinasyon riski, savunma sanayii denetiminde açıklanamazlık). |

---

### 3. Veri Yaklaşımı ve Kalibrasyon

- **Etiketsiz Başlangıç:** Fabrikadaki 1.200 adet etiketsiz görsel için pahalı ön etiketleme yapılmaksızın, gözetimsiz (unsupervised) anomali tespitiyle doğrudan devreye alınabilir yapı kurulmuştur.
- **Temsili Veri Seti Eşlemesi (MVTec AD - CC BY-NC-SA 4.0):** Gerçek savunma sanayii verileri gizlilik nedeniyle paylaşılamadığından endüstriyel standart MVTec AD kullanılmıştır:
  - *Optik lens grubu* $\rightarrow$ `metal_nut` (Kusurlar: çizik, kaplama/renk kusuru)
  - *Termal kamera modülü* $\rightarrow$ `transistor` [Yüksek Risk] (Kusurlar: bükük/kırık bacak, hasarlı gövde)
  - *Gözetleme ünitesi* $\rightarrow$ `cable` (Kusurlar: kablo sıyrığı, montaj/lehim hatası)
- **Conformal Kalibrasyon:** Her kategori için ayrılmış $n=40$ adet sağlam görsel üzerinde leave-one-out mantığıyla anomali skor dağılımı ($s_{cal}$) saklanır. Gelen parçanın istatistiksel $p$-değeri $p = (1 + \sum [s_{cal} \ge s]) / (n_{cal} + 1)$ formülüyle hesaplanır. $p > 0.50$ ve sıfır dedektör tespiti durumunda VLM çağrılmadan **erken çıkış (early exit)** ile KABUL önerisi üretilir.
- **İnsan Kararlarının Etikete Dönüşmesi & YOLO Export:** Operatörün arayüzden onayladığı veya düzelttiği kusur kararları SQLite veri tabanında saklanır. Pano sekmesinden tek tıkla indirilen `yolo_dataset.zip` paketi (görseller + normalize `txt` kutu etiketleri + `classes.txt`) aktif öğrenme döngüsünü besler.
- **SAM ile Ön Etiketleme Yol Haritası:** Biriken 1.200 etiketsiz görsel için anomali ısı haritası pik koordinatları nokta istemi (point prompt) olarak SAM'e (Segment Anything Model) iletilir. Üretilen aday maskeler operatör onayına sunularak etiketleme iş gücü 40 saatten birkaç saate indirilir.

---

### 4. Temel Sistem Mimarisi

```text
[Muayene Görseli] ────────────────────────────────────────────────────────┐
       │                                                                  │
       ▼                                                                  ▼
[Anomali Motoru] ──(Isı haritası + p-değeri)──► [Erken Çıkış?] ◄── [YOLO11 Dedektör]
(PatchCore/EffAD)                                  │ Evet                 │
       │                                           ▼                      │ (Tespit/Kutu)
       │ (Şüpheli / p ≤ 0.50)               [KABUL Önerisi]               │
       └─────────────────────────┬────────────────────────────────────────┘
                                 ▼
                     [Gemini 2.5 Flash VLM ×3] (Isı haritası + YOLO kutuları ipucu)
                                 │ (Tutarlılık + Kusur Tipi + Gerekçe)
                                 ▼
                   [Conformal & Risk Tabanlı Füzyon]
                    ├── KABUL Önerisi   : defect_score < 0.30 & tam sağlam oyu
                    ├── İNSAN İNCELEMESİ: çelişki, sınır skor veya termal risk kuralı
                    └── RET Önerisi     : defect_score ≥ 0.70 & ≥2 kusur oyu
                                 │
                   [Operatör Kararı] ──► [SQLite / AS9100 Kayıt & YOLO ZIP Export]
```

---

### 5. Güvenlik, Denetim ve Savunma Sanayii Uyumluluğu

- **Kimlik Bilgisi ve DoS Koruması:** `GEMINI_API_KEY` kesinlikle kaynak kodda barındırılmaz; ortam değişkeni (`.env`) veya HF Secret üzerinden okunur. Arayüzde `concurrency_limit=2` kuyruk sınırı ile kaynak tükenmesi engellenir.
- **Girdi Doğrulama ve EXIF Temizliği:** Yüklenen dosyalar maksimum 10 MB ile sınırlandırılır. Bellekte PIL ile yeniden kodlanarak potansiyel zararlı baytlar ve konum/kamera bilgisi içeren EXIF üstverileri tamamen temizlenir; en uzun kenar 1024 px'e ölçeklenir.
- **Bulut vs. On-Prem Ayrımı:** PoC aşamasında genel kullanıma açık temsili MVTec verileri harici API'ye iletilmektedir. Gerçek üretimde hiçbir görsel fabrika dışına çıkamaz; sistem tamamen hava boşluklu (air-gapped) yerel ağda, on-prem modellerle çalışmalıdır.
- **AS9100 İzlenebilirlik ve Denetim İzi (Audit Trail):** Her muayene işlemi SQLite veri tabanına değişmezlik ilkesiyle kaydedilir: Orijinal görsel referansı, muayene zamanı, operatör sicili, aktif model sürümleri (`anomaly`, `detector`, `vlm`, `app`), bağımsız motor skorları, birleşik AI önerisi ve operatörün nihai kararı ile açıklaması.
- **İnsan-Döngüde Prensibi (Human-in-the-Loop):** VLM veya AI motorları nihai onay makamı değildir; sistem yalnızca gerekçeli karar desteği sunar. `REVIEW` kararlarında "Onayla" seçeneği kilitlenerek operatör fiziksel incelemeye zorlanır.

---

### 6. Sistemin Sınırlılıkları

1. **Temsili Veri Kısıtı:** Modeller MVTec AD üzerinde yapılandırılmıştır; fabrika ortamındaki gerçek lens yansımalarını, kaplama kırılmalarını ve parça geometrilerini birebir kapsamaz.
2. **2D Görselin Fiziksel Ölçüm Sınırları:** Optik eksen kaçıklığı ve konnektör tork/çekme gevşekliği tek bir 2D fotoğraftan doğrulanamaz. Bu kusurlar için optik kolimatör ve tork ölçüm sensörleri zorunludur (arayüzde bilgi kartı olarak sunulmuştur).
3. **VLM Öz-Güven Kalibrasyonu:** VLM modellerinin metin içi `self_confidence` değerleri aşırı iyimser (overconfident) olabilmektedir. Kod içerisinde bu durum 3 paralel çağrının oy oranı ile çarpılarak ($p_{vlm} = \text{oran} \times \bar{c}$) dengelenmiştir.
4. **Altyapı Kısıtları:** Hugging Face Spaces ücretsiz katmanında kalıcı disk bulunmadığından konteyner yeniden başladığında SQLite sıfırlanır (demo tutarlılığı için açılışta `seed_demo` yüklenir). CPU ortamında PatchCore ve YOLO çıkarımı toplam 1–2 sn gecikme üretir; $n=40$ kalibrasyon boyutu istatistiksel sınır testleri için küçüktür.

---

### 7. Üretime Geçiş Yol Haritası (Production Roadmap)

- **Mekanik Fikstür ve Aydınlatma:** Görsel değişkenliğini minimuma indirmek için sabit muayene fikstürü, telecentric lens ve difüz kubbe (dome) aydınlatma kurulumu.
- **Gerçek Hat Verisi ve Etiketleme:** 1.200 adet saha görselinin toplanması, self-hosted CVAT + SAM destekli ön etiketleme ile hızlı ve güvenli parça kütüphanesi oluşturulması.
- **Model İyileştirme:** Tüm ürün gruplarını tek modelde toplayan Dinomaly mimarisine geçiş; hat verileriyle fine-tune edilmiş YOLO11-seg segmentasyon modeli.
- **On-Premise VLM:** Bulut bağımlılığını ortadan kaldırmak için yerel GPU sunucusunda Qwen2.5-VL (7B/8B) koşturulması; AnomalyR1 yaklaşımıyla GRPO pekiştirmeli öğrenme kullanılarak savunma sanayii muayene muhakemesinin modele kazandırılması.
- **ERP/MES Entegrasyonu:** İş emirlerinin barkod/karekod ile MES'ten otomatik çekilmesi; muayene kararının ve AS9100 denetim kaydının doğrudan kurumsal ERP sistemine aktarılması.
- **MSA / Gage R&R ve Pilot Faz:** Sistem güvenilirliğinin Ölçüm Sistemleri Analizi (Gage R&R) ile doğrulanması; operatörler arası değişkenliğin kanıtlanması amacıyla 2 aylık mevcut süreçle paralel pilot çalışma.
- **MLOps ve Model İzleme:** Veri kayması (data drift) tespiti ve operatörün AI kararını değiştirdiği numunelerden otomatik aktif öğrenme ve yeniden eğitim tetiklenmesi.
- **Maliyet ve Fayda:** Gemini Flash API maliyeti ihmal edilebilir düzeydedir (< ~$100/yıl); asıl ekonomik kazanç 85 parça/günlük muayenede %60–70 uzman süresi tasarrufu ve sıfır kaçan kusur maliyetidir. Üretim fazı için tek seferlik yerel GPU iş istasyonu (örn. RTX 4090 / L40S) yatırımı yeterlidir.

---

### 8. Doğrulama ve Metrik Tablosu

| Bileşen / Model | Hedef Görev | Doğrulama Ölçütü | PoC Ölçülen / Mevcut | Hedeflenen Üretim Değeri |
| :--- | :--- | :--- | :--- | :--- |
| **PatchCore** (`wide_resnet50_2`) | Gözetimsiz Anomali | Image AUROC / Gecikme | [METRİK: PatchCore AUROC] / ~1.2 sn (CPU) | > %99,2 / < 80 ms (GPU) |
| **EfficientAD-S** (ONNX) | Hızlı Anomali & Harita | Image AUROC / Gecikme | [METRİK: EfficientAD AUROC] / [METRİK: CPU ms] | > %98,8 / < 25 ms (GPU) |
| **YOLO11n Dedektör** | Kusur Sınıflandırma | mAP50 / Gecikme | [METRİK: YOLO mAP50] / ~90 ms (CPU) | > %85,0 / < 15 ms (GPU) |
| **Gemini 2.5 Flash VLM** | Muhakeme ve Şiddet | JSON Şema Uyumu / Tutarlılık | %100 Şema Uyumu / [METRİK: VLM Tutarlılık] | > %95 Tutarlılık (On-prem) |
| **Karar Füzyon Kapısı** | Karar Doğruluğu | İnsan İnceleme (Review) Oranı | [METRİK: Review Oranı] | < %15 (Hedef: 85 muayene/gün) |
| **Canlı Prototip** | Uçtan Uca Doğrulama | Dağıtım Durumu | [LİNK: HF Space] | Air-gapped On-Prem Sunucu |

---
*AI destekli geliştirme: kod Claude/Gemini kodlama ajanlarıyla yazıldı, mimari ve kararlar aday tarafından.*
