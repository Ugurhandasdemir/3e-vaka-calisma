# 3E Elektro Optik — AI Destekli Kalite Kontrol Sistemi (PoC)
## Teknik Vaka Dokümanı ve Mimari Rapor

### 1. Problem ve Çözüm Özeti
Günlük 85 adet kritik elektro-optik muayene hacmi, 1.200 adet etiketsiz parça görseli ve ayda 3 kaçan kusurun (özellikle geri çağırma riski yüksek termal modüllerde) yarattığı kalite riski temel operasyonel darboğazdır. Çözüm; gözetimsiz anomali tespiti (PatchCore / EfficientAD-S), nesne dedektörü (YOLO11n) ve görsel dil modelini (Gemini 2.5 Flash) conformal kalibrasyonla birleştiren insan-döngüde (human-in-the-loop) hibrit bir karar destek sistemidir. Sistem uzmanı hızlandırır, kaçan kusurları engeller ve AS9100 uyumlu tam izlenebilirlik sunar.

### 2. Kullanılan AI Modelleri ve Teknolojileri
Sistem **"iki aşamalı sistem"** mimarisidir: Aşama 1 (anomali) etiketsiz bölge önerisi (RPN rolü) üretir; Aşama 2 (YOLO/VLM) bu bölgeleri sınıflandırıp gerekçelendirir.

| Katman | Seçilen Model | Neden Seçildi? | Alternatifler ve Kıyas |
| :--- | :--- | :--- | :--- |
| **Anomali** *(Öneri)* | **PatchCore** (`wide_resnet50_2`) + **EfficientAD-S** (ONNX) | PatchCore sıfır eğitimle yalnız sağlam görsel bankasıyla çalışır (soğuk başlangıç). EfficientAD-S (Colab, ONNX) varsa otomatik kullanılır; CPU ~100–300 ms, GPU 2–3 ms ile F/P lideridir. | Dinomaly (~%99,6 AUROC, ViT-B omurgası CPU için ağır); Zero-shot CLIP (~%90–92 AUROC, ince kusurda zayıf). |
| **Dedektör** *(Sınıflandırma)* | **YOLO11n** (Colab, MVTec maskelerinden etiket) | Hızlı CPU çıkarımı (~90 ms), tek komutla ONNX uyumu ve düşük bellek tüketimi. Bilinen kusurları kutularla sınırlar. | Faster/Cascade/Mask R-CNN (küçük kusurda iyi, CPU 1–3 sn ağır, ONNX ihracı riskli). |
| **VLM** *(Muhakeme)* | **Gemini 2.5 Flash** (Pydantic JSON şema, 3 örnek tutarlılık) | Yapılandırılmış JSON garantisi. Isı haritası ve YOLO kutularını birlikte işler; Myriad (arXiv:2310.19070) uzman ipucu fikrinin eğitimsiz (in-context) görsel uyarlamasıdır. 3 paralel çağrıyla çoğunluk ve tutarlılık ($c/3$) hesaplar. | GPT-4o / Pro (pahalı/yavaş); On-prem Qwen-VL (GPU gerektirir; açık kaynak üretim hedefi). |
| **Füzyon** *(Karar)* | **Conformal $p$-değeri** + Ağırlıklı Skor + 3 Bölge + Risk Kuralı | Skor yerine istatistiksel belirsizlik. Ağırlıklar: Anomali 0.45, Dedektör 0.25, VLM 0.30. Yüksek riskli grupta (*Termal modül*) kusur oyunda otomatik kabul verilmez; doğrudan incelemeye gider. | Saf lojistik regresyon veya kural dışı LLM kararı (halüsinasyon, açıklanamazlık). |

### 3. Veri Yaklaşımı ve Kalibrasyon
- **Etiketsiz Başlangıç:** 1.200 etiketsiz görsel için ön etiketleme gerektirmeden gözetimsiz anomali tespitiyle sıfırıncı günden çalışır.
- **Temsili Veri (MVTec AD - CC BY-NC-SA 4.0):** Optik lens grubu $\rightarrow$ `metal_nut` (çizik, kaplama); Termal kamera modülü $\rightarrow$ `transistor` [Yüksek Risk] (konektör/montaj); Gözetleme ünitesi $\rightarrow$ `cable` (kablo/montaj).
- **Conformal Kalibrasyon:** Ayrılmış sağlam görsellerden ($n=40$) skor dağılımı saklanır; $p = (1 + \sum [s_{cal} \ge s]) / (n_{cal} + 1)$ hesaplanır. $p > 0.50$ ve tespit yoksa VLM atlanarak **erken çıkışla (early exit)** KABUL önerilir.
- **İnsan Kararları & YOLO Export:** Operatör onay/düzeltmeleri SQLite veri tabanına yazılır. Pano sekmesinden indirilen normalize YOLO ZIP paketi (görseller + `.txt` koordinatlar + `classes.txt`) aktif öğrenmeyi besler.
- **SAM Ön Etiketleme Yol Haritası:** 1.200 görselde ısı haritası tepe noktalarından SAM nokta istemiyle otomatik maske/kutu üretilip operatör onayına sunularak etiketleme 40 saatten birkaç saate indirilir.

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
                    ├── KABUL Önerisi   : defect_score < 0.30 & oy birliği
                    ├── İNSAN İNCELEMESİ: çelişki, sınır skor, termal risk kuralı
                    └── RET Önerisi     : defect_score ≥ 0.70 & ≥2 kusur oyu
                                 │
                   [Operatör Kararı] ──► [SQLite / AS9100 Kayıt & YOLO ZIP Export]
```

### 5. Güvenlik, Denetim ve Savunma Sanayii Uyumluluğu
- **Sistem ve API Güvenliği:** `GEMINI_API_KEY` env/secret ile saklanır, koda gömülmez. Yüklemeler 10 MB ile sınırlıdır; PIL ile yeniden kodlanarak zararlı baytlar ve EXIF verileri temizlenir (maks. 1024 px). Gradio `concurrency_limit=2` kuyruk limiti DoS riskini önler.
- **PoC vs. Üretim:** PoC'de temsili MVTec verisi harici API'ye gider. Üretimde hiçbir görsel dışarı çıkamaz; air-gapped on-prem modeller zorunludur.
- **RBAC ve AS9100 İzlenebilirlik:** Rol tabanlı erişim kontrolü (RBAC: operatör, kalite mühendisi, denetçi) ve SQLite denetim kaydı (audit trail): Görsel, model sürümleri (`anomaly`, `detector`, `vlm`, `app`), motor çıktıları/p-değeri, AI önerisi, insan kararı, kullanıcı sicili, zaman damgası ve operatör notu.
- **Karar Otoritesi:** VLM veya AI nihai karar vermez; yalnızca gerekçe sunar. `REVIEW` durumunda "Onayla" kilitlenerek operatör fiziksel incelemeye zorlanır.

### 6. Sistemin Sınırlılıkları
1. **Temsili Veri:** MVTec AD laboratuvar verisidir; gerçek lens yansımalarını ve fabrika ortamını tam kapsamaz.
2. **Fiziksel Ölçüm Sınırları:** Optik eksen hizalama/kaçıklığı ve konektör gevşekliği tek 2D görselden ölçülemez; kolimatör ve tork/çekme testi zorunludur (arayüzde bilgi kartıyla belirtilir).
3. **VLM Kalibrasyonu:** VLM öz-güveni aşırı iyimserdir; 3 paralel çağrının oy oranıyla çarpılarak ($p_{vlm} = \text{oran} \times \bar{c}$) dengelenmiştir.
4. **Altyapı:** Ücretsiz HF Space kalıcı disk sunmaz (`seed_demo` ile başlar); CPU çıkarımı 1–2 sn gecikir; $n=40$ kalibrasyon kümesi istatistiksel sınır testleri için küçüktür.

### 7. Üretime Geçiş Yol Haritası
- **Fikstür ve Aydınlatma:** Sabit fikstür, telecentric lens ve difüz kubbe aydınlatma ile kontrollü görüntüleme.
- **Veri ve Etiketleme:** 1.200 hat görselinin toplanması, self-hosted CVAT + SAM ile hızlı ön etiketleme.
- **Model Geliştirme:** Tüm parçaları kapsayan tek Dinomaly modeli; hat verisiyle fine-tune edilmiş YOLO11-seg.
- **On-Premise VLM:** Bulut bağımlılığını kaldıran yerel GPU sunucusunda Qwen2.5-VL (7B/8B) koşturulması; AnomalyR1 tarzı GRPO pekiştirmeli öğrenme ile muayene muhakemesinin eğitilmesi.
- **ERP/MES Entegrasyonu:** İş emri çekme, seri no eşleme ve muayene sonucunun ERP'ye CSV/API ile aktarımı.
- **MSA / Gage R&R ve Pilot:** Ölçüm sistemleri analizi (Gage R&R) ile operatör varyansı doğrulaması; 2 aylık paralel pilot çalışma.
- **Model İzleme (MLOps):** Veri kayması tespiti ve değiştirilen kararlardan otomatik yeniden eğitim döngüsü.
- **Maliyet / Yatırım:** Gemini API maliyeti ihmal edilebilir (< ~$100/yıl); asıl kazanç 85 muayene/günde %60–70 uzman süresi tasarrufu ve sıfır kaçan kusurdur. Üretim için tek seferlik on-prem GPU iş istasyonu (RTX 4090 / L40S) yeterlidir.

### 8. Doğrulama ve Metrik Tablosu

| Model / Katman | Doğrulama Ölçütü | PoC Ölçülen / Durum | Hedef Üretim Değeri |
| :--- | :--- | :--- | :--- |
| **PatchCore** (`wide_resnet50_2`) | Image AUROC / Gecikme | [METRİK: PatchCore AUROC] / ~1.2 sn (CPU) | > %99,2 / < 80 ms (GPU) |
| **EfficientAD-S** (ONNX) | Image AUROC / Gecikme | [METRİK: EfficientAD AUROC] / [METRİK: CPU ms] | > %98,8 / < 25 ms (GPU) |
| **YOLO11n Dedektör** | mAP50 / Gecikme | [METRİK: YOLO mAP50] / ~90 ms (CPU) | > %85,0 / < 15 ms (GPU) |
| **Gemini 2.5 Flash VLM** | JSON Şema Uyumu / Tutarlılık | %100 Şema Uyumu / [METRİK: VLM Tutarlılık] | > %95 Tutarlılık (On-prem) |
| **Füzyon Karar Kapısı** | İnsan İnceleme (Review) Oranı | [METRİK: Review Oranı] | < %15 (85 muayene/gün) |
| **Canlı Prototip** | Dağıtım Durumu | [LİNK: HF Space] | On-Prem Air-gapped Küme |

---
*AI destekli geliştirme: kod Claude/Gemini kodlama ajanlarıyla yazıldı, mimari ve kararlar aday tarafından.*
