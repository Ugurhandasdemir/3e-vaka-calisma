---
title: 3E Vaka — Mimari Kararlar ve Yöntem Özetleri
created: 2026-10-03
modified: 2026-10-03
type: project
status: active
tags: [3e-elektro-optik, vaka-calismasi, mimari, anomali-tespiti, vlm, yolo, sam]
---

# 3E Vaka — Mimari Kararlar ve Yöntem Özetleri

Bağlam: [[Hibrit-Sistem-Arastirma]] (saha, maliyet, literatür). Bu not 2026-10-03 vaka sırasındaki
karar konuşmasının özeti. Benchmark sayıları Claude'un bilgisinden, **yaklaşık, doğrulanmadı**.

## 1. Model seçimi (başarı vs fiyat/performans)

### Anomali tespiti (MVTec AD görsel AUROC)
| Model                                 | Başarı                          | Hız / maliyet               | Not                                             |
| ------------------------------------- | ------------------------------- | --------------------------- | ----------------------------------------------- |
| Dinomaly (CVPR 2025)                  | ~%99,6 çok sınıflı; VisA ~%98,7 | ViT-B DINOv2, ağır          | **En yüksek başarı**, tek model tüm kategoriler |
| PatchCore                             | ~%99,1–99,6 sınıf başına        | eğitim yok, CPU ~0,5–1 sn   | Kurulumu en kolay                               |
| EfficientAD-S                         | ~%98,8 (M ~%99,1)               | GPU 2–3 ms, CPU ~100–300 ms | **F/P lideri**, Colab ~30 dk eğitim             |
| Zero-shot CLIP (AnomalyCLIP, AA-CLIP) | ~%90–92                         | eğitim yok                  | Gözetimsizden ~7 puan geride                    |

MVTec AD 2 (şeffaf/yansıtıcı) tüm yöntemlerde ciddi düşüş: lens için gerçekçi ölçüt.

### Dedektör
- F/P: **YOLO11s(-seg)**. En yüksek: RF-DETR / D-FINE (ağır, PoC'ye gerekmez).

### VLM (MMAD, ICLR 2025)
- En iyi GPT-4o ~%75 ortalama, insan uzman altında; ince kusur ve lokalizasyonda zayıf.
- AnomalyR1 / IAD-R1: küçük Qwen2.5-VL + GRPO ile büyükleri geçme iddiası (doğrulanmadı).
- Güncel frontier (Gemini 2.5/3, GPT-5) MMAD skorları doğrulanmadı.
- PoC F/P: **Gemini Flash** (ucuz, ücretsiz katman, JSON şema, kutu çıktısı). En yüksek: Pro sınıfı.
  On-prem: Qwen-VL 7–8B + GRPO.

## 2. Neden iki aşamalı dedektör (Faster/Cascade/Mask R-CNN) değil
1. HF CPU'da ~1–3 sn/görsel; YOLO11s ~100 ms.
2. YOLO tek komut ONNX; Detectron2/MMDet export riskli.
3. Genel benchmark'ta fark kapandı.

İki aşamalının haklı olduğu yer: küçük/ince kusur (Cascade R-CNN), Mask R-CNN maskesiyle çizik ölçümü
(MIL-PRF-13830B scratch-dig). YOLO karşılığı: SAHI tiling, YOLO11-seg.

**Sistem zaten iki aşamalı:** Aşama 1 anomali ısı haritası = etiketsiz bölge önerisi (RPN rolü),
Aşama 2 YOLO/VLM = sınıflandırma. Sunum cümlesi: "İki aşamalı dedektör yerine iki aşamalı sistem;
öneri etiketsiz, sınıflandırma etiketli." Zaman kalırsa Colab'da Faster R-CNN kıyas tablosu (mAP, gecikme).

## 3. SAM ile etiketleme
- **1.200 etiketsiz görsel:** ısı haritası tepe noktası → SAM nokta istemi → maske → kutu → insan
  doğrular → YOLO eğitimi. Yol haritası: SAM 3 metin istemi ("scratch"). Etiketleme ~25–40 saatten
  birkaç saatlik doğrulamaya.
- **Uygulama içi:** kullanıcı düzeltirken kusura tıklar → MobileSAM / SAM 2.1-tiny (CPU ~1 sn) → hassas etiket.
- MVTec eğitiminde gerek yok (maskeler hazır).
- Uyarı: düşük kontrastlı ince çizikte zayıf; nokta istemi + insan doğrulama şart.

## 4. Akış şeması

```
═══ ÇEVRİMDIŞI (Colab T4, Uğurhan) ═══
 MVTec AD (temsili veri)
   ├─ sağlam görseller ─► EfficientAD-S ─► anomaly.onnx + calib.npy
   └─ kusur maskeleri ─► maske→kutu/poligon ─► YOLO11s-seg ─► yolo.onnx
                         (ops.: Faster R-CNN kıyas)

═══ ÇEVRİMİÇİ (HF Spaces, Gradio, CPU) ═══
 [1] Kullanıcı: ürün grubu + seri no + görsel
 [2] Aşama 1: EfficientAD (yedek PatchCore) → skor + ısı haritası → conformal p
       └─ açık sağlam → VLM'siz KABUL önerisi
 [3] Aşama 2: YOLO11s-seg (sınıf, kutu, maske)  ║  Gemini Flash (görsel + ısı haritası
       + YOLO kutuları → JSON: kusur?, tip, şiddet, gerekçe; ×3 → tutarlılık)
 [4] Füzyon: w1·anomali_p + w2·YOLO + w3·VLM_tutarlılık
       → KABUL önerisi | İNSAN İNCELEMESİ (çelişki/sınır) | RET önerisi
 [5] Sonuç ekranı: öneri, güven, ısı haritası, kutu/maske, gerekçe, motor dökümü
 [6] Kullanıcı kararı: Onayla / Değiştir / Not  (+ tıkla → SAM maske)
 [7] Kayıt (AS9100): SQLite → geçmiş + pano, CSV (ERP tek giriş), YOLO etiket export

═══ GERİ BESLEME ═══
 insan düzeltmeleri + SAM maskeleri ─► etiketli havuz ─► Colab yeniden eğitim ─► yeni .onnx
 1.200 etiketsiz ─► ısı haritası noktası ─► SAM ön etiket ─► insan doğrular ─┘
```

Kapsam dışı (sınırlılık): optik eksen hizalaması (kolimatör), konektör gevşekliği (tork/çekme testi).

İş bölümü: Uğurhan Colab eğitimi; agy/Sonnet Gradio uygulaması (önce PatchCore yedekle); Claude spec,
inceleme, 2 sayfa doküman.

## 5. Myriad ve alternatif VLM+uzman yöntemleri

**Myriad (arXiv:2310.19070):** VLM ince kusuru göremez, yanına uzman göz eklenir. Uzman (PatchCore/AprilGAN)
anomali haritası → Expert Perception modülü haritayı token'a çevirir → görsel token'larla birlikte LLM'e
(MiniGPT-4). Vision Expert Instructor encoder'ı şüpheli bölgeye odaklar. Bedeli: modül + LLM kısmı eğitilir.

**Bizimki Myriad değil, eğitimsiz görsel istemleme:** ısı haritası görsele bindirilir, YOLO kutuları
prompt'a yazılır, Gemini'ye gider. Sunum cümlesi: "Myriad'ın uzman ipucu fikrini eğitimsiz, API ile uyguladım."

| Yöntem                     | Yıl            | Yaklaşım                                                                        | Artı                             | Eksi                                    |
| -------------------------- | -------------- | ------------------------------------------------------------------------------- | -------------------------------- | --------------------------------------- |
| AnomalyGPT                 | 2023 (AAAI'24) | görsel-metin benzerliği → harita, prompt learner → Vicuna; sahte kusurla eğitim | few-shot, lokalizasyon           | eğitim, 7B ağır                         |
| Myriad                     | 2023           | dış uzman haritası → token → LLM                                                | uzman değiştirilebilir           | eğitim                                  |
| Anomaly-OV                 | 2025 (CVPR)    | "look-twice" sağlam/şüpheli kıyas, Anomaly-Instruct-125k                        | zero-shot tespit + gerekçe       | eğitim (ayrıntı doğrulanmadı)           |
| AnomalyR1 / IAD-R1         | 2025           | uzmansız, küçük Qwen2.5-VL + GRPO                                               | küçük model güçlü, on-prem uygun | eğitim verisi + GPU (skor doğrulanmadı) |
| Görsel istemleme (bizimki) | —              | harita bindirme, kırpıntı, kutu koordinatı                                      | eğitimsiz, her API               | token düzeyi aktarım yok                |
| Saf VLM (MMAD)             | 2025           | ham görsel + soru                                                               | en basit                         | ince kusurda zayıf (~%75)               |

**Yol haritası:** on-prem Qwen-VL + AnomalyR1 tarzı GRPO + Myriad tarzı uzman girdisi.

Okuma düzeyi: Myriad, AnomalyGPT, MMAD, AnomalyR1 yöntem özeti düzeyinde; Anomaly-OV ve IAD-R1 ayrıntıları
doğrulanmadı, sunumda sayı verme.
