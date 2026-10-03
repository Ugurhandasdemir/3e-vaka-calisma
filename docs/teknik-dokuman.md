# 3E Elektro Optik — AI Kalite Kontrol PoC Raporu
Canlı: [HF Spaces](https://huggingface.co/spaces/ugurhandasdemir/3e-vaka-calisma) (ZeroGPU, CPU) | Kod: [GitHub](https://github.com/Ugurhandasdemir/3e-vaka-calisma)

### 1. Problem ve Çözüm Özeti
Günlük 85 kritik muayene, 1.200 etiketsiz görsel ve 6 ayda müşteriye ulaşan 3 kusur (biri termal modül geri çağırması) mevcut manuel sürecin sınırlarını gösteriyor. Çözüm; anomali tespiti (PatchCore / EfficientAD-S), dedektör (YOLO11n) ve VLM'i (Gemini 3.5 Flash-Lite) conformal kalibrasyonla birleştiren insan-döngüde hibrit karar desteğidir. Sistem uzmanı hızlandırır, AS9100 uyumlu izlenebilirlik sunar. Canlı gecikme VLM ile ~4–6 sn, erken çıkışta ~2–3 sn'dir. Hibrit kazanımı: VLM'in tek başına kaçırdığı `misplaced` transistör montaj kusuru, anomali skoru ve risk kuralıyla yakalanıp insan incelemesine (`REVIEW`) yönlendirilmiştir.

### 2. Kullanılan AI Modelleri ve Teknolojileri
Aşama 1 (anomali) bölge önerir; Aşama 2 (YOLO/VLM) sınıflandırıp gerekçelendirir.

| Katman | Seçilen Model ve Gerekçe | Alternatifler |
| :--- | :--- | :--- |
| **Anomali** *(Öneri)* | **PatchCore** (`wide_resnet50_2`, 60 ref, %10 coreset, CPU ~120–150 ms) + **EfficientAD-S** (ONNX, Colab T4 metal_nut AUROC 0.9868, 80 dk). Eğitimsiz hızlı başlangıç; ONNX varsa otomatik devrededir. | Dinomaly (ViT-B CPU'da ağır); CLIP (lokalde zayıf). |
| **Dedektör** *(Sınıflandırma)* | **YOLO11n** (ONNX CPU ~50 ms). 10 MVTec sınıfı gerçek kusurlarından 3 sınıfa Colab'da eğitildi (~1.000 artırılmış veri). Kusurları kutular. | Faster R-CNN (CPU 1–3 sn, ONNX hantal). |
| **VLM** *(Muhakeme)* | **Gemini 3.5 Flash-Lite** (yedek: 3.8-Flash, 3.1-Flash-Lite; ~2 sn). JSON şemalı; görsel, ısı haritası ve YOLO kutularını Myriad tarzı ipucu alır; 3 paralel örnekle çoğunluk/tutarlılık üretir. | GPT-4o (maliyet); On-prem Qwen-VL (PoC'de API esnekliği). |
| **Füzyon** *(Karar)* | **Conformal $p$ + Ağırlıklı Skor + Risk Kuralı**. Ağırlık: Anomali 0.45, Dedektör 0.25, VLM 0.30. Termal modülde kusur oyunda otomatik kabul engellenir. | Bağımsız LLM kararı; Lojistik regresyon. |

### 3. Veri Yaklaşımı ve Kalibrasyon
- **Temsili Veri (MVTec AD - CC BY-NC-SA 4.0):** Optik lens $\rightarrow$ `metal_nut`, Termal kamera $\rightarrow$ `transistor` (Yüksek Risk), Gözetleme ünitesi $\rightarrow$ `cable`.
- **Dedektör:** YOLO11n; 10 MVTec sınıfından gerçek kusur maskelerinden kutu üretilip çevrimdışı artırımla ~1.000 eğitim görseliyle Colab'da eğitildi (3 uygulama + vida, kapsül, hap, şişe, fındık, ahşap, fayans; val yalnızca gerçek kusurlar).
- **Conformal Kalibrasyon:** 40 sağlam görselle $p = (1 + \sum [s_{cal} \ge s]) / (n_{cal} + 1)$ hesaplanır. Anomali olasılığı $a = \text{clip}(\log p / \log 0.02, 0, 1)$; $a \ge 0.75$ ($p \le \sim 0.05$) kusur oyudur.
- **Erken Çıkış (Early Exit):** $p > 0.50$ ve tespit yoksa VLM atlanıp gecikme ~2–3 sn'ye iner; termal kamera erken çıkış almaz, tüm motorlar çalışır.
- **Geri Bildirim:** Kararlar SQLite'a yazılır; indirilen YOLO formatlı ZIP aktif öğrenmeyi besler.

### 4. Temel Sistem Mimarisi
```text
[Görsel] ──► [Anomali: PatchCore/EffAD] ──(Isı haritası + p)──┐
         ──► [Dedektör: YOLO11n] ────────(Kutular)─────────────┼─► [Erken Çıkış? (p>0.5, yok)]
                                                               │    ├── Evet: [KABUL]
                                                               └──► └── Hayır: [Gemini 3.5 ×3]
                                                                                   │
[Karar Füzyonu] ◄──────────────────────────────────────────────────────────────────┘
 ├── KABUL Önerisi   : skor < 0.30 & tüm motorlar sağlam
 ├── RET Önerisi     : skor ≥ 0.70 & ≥2 kusur oyu
 └── İNSAN İNCELEMESİ: çelişki / sınır skor / termal risk / tek motor
```
*Güven = $\text{tutarlılık} \times \max(\text{skor}, 1 - \text{skor})$. Tek motor asla otomatik karar veremez.*

### 5. Güvenlik, Denetim ve Savunma Sanayii Uyumluluğu
- **API ve Hijyen:** `GEMINI_API_KEY` ortam değişkenindedir. Yüklemeler 10 MB ile sınırlı; görsel RGB'ye çevrilip en fazla 1024 px'e küçültülür (VLM'e 768 px gider), meta veri taşınmaz. Gradio kuyruğu eşzamanlı işi 2 ile sınırlar.
- **Savunma Sanayii:** PoC'de veri Google API'ye çıkar; üretimde savunma gizliliği gereği veri dışarı çıkamaz, sistem hava boşluklu (air-gapped) yerel GPU'da çalışır.
- **AS9100 İzlenebilirlik:** Görsel, model sürümleri (`anomaly`, `detector`, `vlm`, `app`), $p$-değeri, AI önerisi, nihai karar, muayeneci adı ve zaman damgası SQLite denetim izinde (audit trail) saklanır.
- **Karar Otoritesi:** AI karar vermez, önerir. İnceleme (`REVIEW`) sonucunda "AI önerisini onayla" seçeneği kapanır; uzman Kabul veya Ret seçmek zorundadır.

### 6. Sistemin Sınırlılıkları
1. **Veri Örtüşmesi:** MVTec test kümesi YOLO eğitim/artırımında kullanıldığından demo örneklerinde örtüşme (overlap) olabilir; laboratuvar koşulları fabrika parlamalarını tam içermez.
2. **Fiziksel Ölçüm:** Optik eksen kaçıklığı ve konektör gevşekliği tek 2D fotoğraftan ölçülemez; kolimatör ve tork/çekme aparatıyla fiziksel ölçüm şarttır (arayüzde uyarılır).
3. **VLM Kalibrasyonu:** VLM öz-güveni (`self_confidence`) kalibre değildir; çoğunluk oranıyla ($p_{vlm} = \text{oran} \times \bar{c}$) dengelenmiştir.
4. **Altyapı:** HF Spaces geçici disk (ephemeral) kullanır; veriler sıfırlanabilir. PoC'de veri Google API'ye çıkar; kalibrasyon 40 görselle sınırlıdır.

### 7. Üretime Geçiş Yol Haritası
- **On-Premise Modeller:** Dinomaly veya INP-Former ile anomali; fabrika verisiyle ince ayarlı (fine-tuned) YOLO segmentasyon.
- **Yerel VLM:** Veri gizliliği için yerel GPU'da AD-Copilot veya IAD-R1 mimarisinde kompakt VLM (ör. Qwen2.5-VL).
- **Referans Kıyaslama:** Sabit fikstür, telecentric lens ve difüz aydınlatma; şüpheli parçayı referans numuneyle doğrudan kıyaslama.
- **SAM Ön Etiketleme:** 1.200 görselin SAM ile otomatik etiketlenip uzman kontrolüne sunulması.
- **Metroloji ve Pilot:** MSA / Gage R&R ile operatör varyansı analizi; mevcut manuel süreçle paralel gölge pilot (öneri).
- **ERP / MES Entegrasyonu:** İş emri ve seri no eşleşmesi; muayene kararının ERP/MES'e kalite kapısı olarak aktarımı.

### 8. Doğrulama ve Metrik Tablosu

| Bileşen / Model | Ölçülen Metrik (PoC) | Ortam / Detay |
| :--- | :--- | :--- |
| **PatchCore** (`wide_resnet50_2`) | Gecikme: ~120–150 ms/görsel | CPU, 60 sağlam ref, %10 coreset |
| **EfficientAD-S** (ONNX) | Image AUROC: 0.9868 (metal_nut) | Colab T4 (80 dk), varsa otomatik devrede |
| **YOLO11n Dedektör** | mAP50: 0.814, mAP50-95: 0.518 | ONNX CPU ~50 ms (10 MVTec sınıfı, ~1.000 veri) |
| **YOLO11n Hedef Sınıflar** | mAP50: 0.779 (3 hedef kategori) | scratch: 0.813, coating: 0.839, conn: 0.790 |
| **Gemini 3.5 Flash-Lite** | Çıkarım: ~2.0 sn, 3 paralel örnek | Pydantic JSON, Myriad görsel yönlendirme |
| **Uçtan Uca Boru Hattı** | ~4–6 sn (Erken çıkış: ~2–3 sn) | Canlı HF Spaces (ZeroGPU, CPU çıkarımı) |
| **Füzyon Kuralı** | Ağırlık: 0.45 Anom, 0.25 Ded, 0.30 VLM | Kabul: <0.30, Ret: ≥0.70 & ≥2 kusur oyu |

---
*AI destekli geliştirme: kod Claude/Gemini kodlama ajanlarıyla yazıldı, mimari ve kararlar aday tarafından.*
