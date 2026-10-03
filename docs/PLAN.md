---
title: 3E Vaka — Uygulama Planı (PoC)
created: 2026-10-03
modified: 2026-10-03
type: project
status: active
tags: [3e-elektro-optik, vaka-calismasi, plan, poc, gradio, hf-spaces]
---

# 3E Vaka — Uygulama Planı (PoC)

Dayanak: [[Mimari-Kararlar]], [[Hibrit-Sistem-Arastirma]]. Uygulama "başla" komutundan sonra.

## 0. Hedef ve teslimler
| Teslim | Ne | Nerede |
|---|---|---|
| 1. Yayınlanmış uygulama | Gradio, HF Spaces (CPU basic, ücretsiz) | `https://huggingface.co/spaces/<kullanıcı>/3e-qc-poc` |
| 2. Kaynak kod | GitHub public repo (+ HF Space aynı kod) | `github.com/Ugurhandasdemir/3e-qc-poc` |
| 3. Teknik doküman | ≤2 sayfa PDF/MD | repo `docs/teknik-dokuman.md` + PDF |

Temel akış (vaka şartı): **görsel yükleme → AI analizi → sonuç → güven skoru → kullanıcı kararı.**

## 1. Kesinleşen kararlar
| Konu | Karar |
|---|---|
| Anomali | EfficientAD-S (anomalib, Colab), ONNX; yedek PatchCore (ResNet18, eğitimsiz, açılışta referanslardan) |
| Dedektör | YOLO11s-seg (Colab, MVTec maskeleri → poligon), ONNX; yoksa motor devre dışı, ağırlık yeniden dağıtılır |
| VLM | Gemini Flash (`GEMINI_MODEL` env ile değiştirilebilir), JSON şema, ısı haritası bindirme + YOLO kutuları ipucu, 3 çağrı tutarlılık |
| Füzyon | Conformal p + ağırlıklı skor + 3 bölge (KABUL / İNSAN / RET) |
| Etiketleme | MobileSAM tıkla-maskele (zaman kalırsa), SAM toplu ön etiket yol haritasında |
| Kayıt | SQLite + CSV export + YOLO etiket export |
| Hosting | HF Spaces CPU; secret `GEMINI_API_KEY` |
| Veri | MVTec AD (CC BY-NC-SA 4.0), "temsili veri" diye etiketlenir |
| Kod klasörü | `/mnt/windows/linux/Python/3e-qc-poc/` |

### Veri eşlemesi (temsili)
| Ürün grubu (uygulamada) | MVTec AD kategorisi | Kusur → bizim sınıf |
|---|---|---|
| Optik lens grubu | `metal_nut` | scratch → Yüzey çiziği; color → Kaplama kusuru |
| Termal kamera modülü | `transistor` | bent_lead / cut_lead / misplaced → Konektör/montaj kusuru; damaged_case → Yüzey çiziği |
| Gözetleme ünitesi | `screw` | scratch_head / scratch_neck → Yüzey çiziği; manipulated_front / thread_* → Montaj kusuru |

Optik eksen hizalama: görselden tespit edilmez; uygulamada "ölçüm verisi gerekir" bilgi kartı.
**Açık onay:** kategori seçimi (alternatif: daha az kategori = daha hızlı eğitim).

## 2. Repo yapısı
```
3e-qc-poc/
├─ app.py                 # Gradio Blocks UI (Lane C)
├─ pipeline.py            # motorları çağırır, füzyon, sonuç dict (Lane B)
├─ engines/
│  ├─ anomaly.py          # EfficientAD ONNX + PatchCore yedek, ısı haritası (Lane B)
│  ├─ detector.py         # YOLO11s-seg ONNX (ultralytics) (Lane B)
│  ├─ vlm.py              # Gemini, şema, overlay, ×3 (Lane B)
│  └─ sam.py              # MobileSAM tıkla-maskele (opsiyonel, Lane C)
├─ fusion.py              # skor, bant, gerekçe (Opus spec, Lane B yazar)
├─ calib.py               # conformal p-değeri
├─ storage.py             # SQLite, CSV, YOLO export (Lane C)
├─ config.py              # sınıflar, eşlemeler, ağırlıklar, eşikler
├─ models/                # anomaly_<kat>.onnx, calib_<kat>.npy, yolo_seg.onnx (LFS)
├─ samples/<kat>/{good,defect}/   # referans + demo görselleri
├─ notebooks/01_efficientad.ipynb, 02_yolo_seg.ipynb, (03_sam_autolabel.ipynb)
├─ tests/smoke_test.py    # VLM mock'lu uçtan uca test
├─ docs/teknik-dokuman.md
├─ requirements.txt, README.md (HF Space kartı), .env.example, .gitignore
```

## 3. Bileşen spesifikasyonları

### 3.1 Anomali (`engines/anomaly.py`)
- `AnomalyEngine(category).predict(img) -> {score, heatmap(HxW 0-1), p_value, backend}`.
- ONNX varsa onnxruntime ile EfficientAD (256×256, anomalib normalizasyonu). Yoksa PatchCore yedeği:
  ResNet18 layer2+layer3, 3×3 ortalama havuz, `samples/<kat>/good` görsellerinden hafıza bankası,
  rastgele %10 coreset, kNN (k=1) mesafe → ısı haritası (Gauss blur, orijinal boyuta).
- `calib.py`: `p = (1 + #{s_cal ≥ s}) / (n_cal + 1)`; `calib_<kat>.npy` yoksa sağlam referanslarda
  leave-one-out skorlarından üretilir.

### 3.2 Dedektör (`engines/detector.py`)
- `ultralytics.YOLO("models/yolo_seg.onnx", task="segment")`, conf 0.25, imgsz 640.
- Çıktı: `[{cls, label_tr, conf, box, polygon}]` + çizimli görsel. Model yoksa `available=False`.

### 3.3 VLM (`engines/vlm.py`)
- `google-genai` SDK, `response_schema` (pydantic):
  `{defect_present: bool, defect_type: enum[Yüzey çiziği, Kaplama kusuru, Konektör/montaj kusuru,
  Optik eksen şüphesi, Diğer, Yok], severity: enum[Düşük, Orta, Yüksek], location: str,
  reasoning: str (TR, ≤3 cümle), self_confidence: 0-1}`.
- Girdi: orijinal görsel + ısı haritası bindirilmiş görsel + metin: ürün grubu, anomali skoru/p,
  YOLO kutuları (sınıf, güven, koordinat). Sistem prompt'u: "kalite muayene uzmanı, ipuçları yanılabilir,
  görmediğini uydurma".
- 3 çağrı paralel (temperature 0.4) → çoğunluk tipi, `consistency = çoğunluk/3`.
- Zaman aşımı 20 sn; hata/anahtar yok → `available=False`, akış devam eder (degrade).

### 3.4 Füzyon (`fusion.py`) — Opus spec
- Motor olasılıkları: `a = 1 − p_value`; `y = max YOLO conf` (yoksa 0); `v = (defect oyu oranı) × ort. self_confidence`.
- `defect_score = Σ wᵢ·xᵢ / Σ wᵢ (yalnız mevcut motorlar)`; ağırlıklar `a 0.45, y 0.25, v 0.30`.
- Oylar: `a ≥ 0.95`, `y ≥ 0.40`, `v ≥ 0.5` → kusur oyu. `agreement = çoğunluk oy oranı`.
- Bantlar:
  - **KABUL önerisi:** `defect_score < 0.30` ve tüm motorlar "sağlam".
  - **RET önerisi:** `defect_score ≥ 0.70` ve ≥2 motor "kusur".
  - **İNSAN İNCELEMESİ:** diğer her durum (çelişki, sınır skor, tek motor).
  - Risk kuralı: ürün grubu "Termal kamera modülü" ve herhangi bir motor kusur derse en az İNSAN
    (geri çağırma geçmişi gerekçesiyle).
- Güven skoru (UI): `confidence = agreement × max(defect_score, 1 − defect_score)`, % olarak.
- Kusur tipi: VLM çoğunluk tipi; VLM yoksa en güvenli YOLO sınıfı.
- Erken çıkış: `p_value > 0.5` ve YOLO boş → VLM çağrılmaz, KABUL önerisi (maliyet/gecikme).
- Çıktıya insan okunur gerekçe listesi (`["Anomali p=0.01 (yüksek)", "YOLO: çizik %82", ...]`).

### 3.5 UI (`app.py`)
- **Sekme 1 — Muayene:** sol: ürün grubu, seri no, muayeneci adı, görsel yükle, örnek galerisi,
  [Analiz et]. Sağ: karar rozeti (yeşil/sarı/kırmızı), güven %, ısı haritası, YOLO çizimi, VLM gerekçesi,
  motor döküm tablosu, gecikme. Alt: karar radyo (Onayla / Değiştir), kusur tipi seçimi, not, [Kaydet].
- **Sekme 2 — Geçmiş ve Pano:** tablo, filtre, CSV indir, YOLO etiket ZIP indir; metrikler: toplam muayene,
  AI-insan uyumu %, insan incelemesine düşen %, kusur tipi dağılımı (bar).
- **Sekme 3 — Etiketleme (opsiyonel):** görsel + tıkla → MobileSAM maske → sınıf seç → havuza ekle.
- **Sekme 4 — Hakkında:** mimari şema, sınırlılıklar, veri lisansı notu.
- Türkçe arayüz, mobilde kullanılabilir.

### 3.6 Kayıt (`storage.py`)
- Tablo `inspections`: id, ts, inspector, product_group, serial_no, image_path, model_versions (json),
  engine_outputs (json), ai_decision, ai_defect_type, confidence, human_decision, human_defect_type,
  note, sam_mask_path.
- HF ücretsiz katmanda disk kalıcı değil → dokümanda sınırlılık; açılışta demo kayıtları seed edilir.

## 4. Colab notebook'ları (Uğurhan çalıştırır)
**01_efficientad.ipynb** (T4, kategori başına ~20–30 dk)
1. `pip install anomalib` (sürüm sabitlenir), MVTec AD kategori indirme (anomalib datamodule).
2. EfficientAD-S, 256 px, `max_steps` ~3000 (süreye göre), imagenette otomatik.
3. Test AUROC / F1 yazdır (dokümana girer).
4. ONNX export + sağlam doğrulama görsellerinden `calib_<kat>.npy`.
5. `models/` ZIP indir.

**02_yolo_seg.ipynb** (~30–45 dk)
1. MVTec `ground_truth` maskeleri → kontur → YOLO-seg poligon, sınıf eşlemesi (bölüm 1 tablosu).
2. Train/val %80/20 (test kusurluları), augment (flip, HSV).
3. `yolo11s-seg.pt` fine-tune, imgsz 640, ~60 epoch (erken durdurma).
4. mAP50 / mAP50-95 yazdır, ONNX export.
5. (Ops.) Faster R-CNN torchvision kıyas: mAP + gecikme tablosu.

**03_sam_autolabel.ipynb** (opsiyonel): ısı haritası tepe noktası → SAM → maske → kutu, görsel karşılaştırma.

## 5. İş bölümü (orkestrasyon)
| Lane | Kim | OWNS | Kabul kontrolü |
|---|---|---|---|
| A — Notebook'lar | agy (Gemini) | `notebooks/` | Opus okuma incelemesi; Uğurhan Colab'da hatasız koşar |
| B — Motorlar + füzyon | Sonnet | `engines/anomaly.py, detector.py, vlm.py`, `pipeline.py`, `fusion.py`, `calib.py`, `config.py`, `tests/` | `python tests/smoke_test.py` (VLM mock) geçer; örnek sağlam → KABUL, kusurlu → RET/İNSAN |
| C — UI + kayıt + deploy dosyaları | agy (Gemini) | `app.py`, `storage.py`, `engines/sam.py`, `README.md`, `requirements.txt` | `python app.py` lokal açılır, akış uçtan uca tamamlanır |
| D — Örnek veri | agy | `samples/` | kategori başına ≥15 good + ≥5 defect |
| Opus | Claude | spec, inceleme, entegrasyon, doküman taslağı | kabul kontrollerini kendi koşar |
| Eğitim, hesaplar, demo | Uğurhan | Colab, HF/GitHub token, sunum | — |

Kontrat: `pipeline.run(image, product_group) -> dict` (Lane B) — Lane C yalnız bunu çağırır.

## 6. Zaman çizelgesi (başla anından itibaren)
| Süre | İş | Kritik yol |
|---|---|---|
| 0:00–0:20 | Lane A notebook'lar; repo iskeleti, `config.py`, kontrat | ✔ Uğurhan'ın eğitimi bekliyor |
| 0:20–1:30 | Uğurhan Colab eğitimi ∥ Lane B + C + D | |
| 1:30–2:00 | ONNX entegrasyonu, smoke test, lokal demo | ✔ |
| 2:00–2:30 | HF Space + GitHub yayın, secret, soğuk açılış testi, "değerlendirici gözüyle" deneme | ✔ |
| 2:30–3:15 | Teknik doküman (Opus taslak, Uğurhan düzenler), README | |
| 3:15–4:00 | Tampon, SAM sekmesi (vakit varsa), demo provası | |

## 7. Yedek planlar
| Risk | Yedek |
|---|---|
| Colab GPU yok / eğitim yetişmez | PatchCore yedeği + YOLO devre dışı; akış yine tam çalışır |
| Gemini kota/hata | VLM devre dışı mesajı, füzyon kalan motorlarla; opsiyonel Claude/OpenAI anahtarı |
| HF build hatası / RAM | `opencv-python-headless`, CPU torch wheel, model boyutu küçük; son çare ev sunucusu + tünel |
| Soğuk açılış yavaş | Demo öncesi Space'i uyandır; README'de not |

## 8. Güvenlik (dokümana da girer)
- API anahtarı yalnız HF secret / env; repoda `.env.example`.
- Yükleme doğrulama: yalnız JPG/PNG, ≤10 MB, yeniden boyutlandırma; EXIF temizleme.
- Gradio kuyruk + eşzamanlılık limiti (kötüye kullanım/maliyet).
- PoC'de görseller Google API'ye gider → yalnız temsili veri; üretimde on-prem VLM, ağ izolasyonu,
  RBAC, değiştirilemez denetim kaydı (AS9100), model sürüm kaydı.

## 9. Teknik doküman iskeleti (≤2 sayfa)
1. Problem ve çözüm özeti (3 satır)
2. AI teknolojileri: EfficientAD/PatchCore, YOLO11s-seg, Gemini Flash, conformal füzyon, SAM — neden (F/P, literatür)
3. Veri yaklaşımı: etiketsiz başlangıç (iyi-only), MVTec temsili veri, aktif öğrenme + SAM ile 1.200 görsel
4. Mimari: akış şeması (küçük)
5. Güvenlik
6. Sınırlılıklar: temsili veri, optik eksen/konektör ölçüm gerektirir, VLM güveni kalibre değil, kalıcı disk yok
7. Üretime geçiş: on-prem GPU, gerçek veri + etiketleme, Dinomaly/fine-tune YOLO, on-prem Qwen-VL (GRPO),
   ERP/MES entegrasyonu, kontrollü aydınlatma/fikstür, MSA (gage R&R) doğrulaması, model izleme
8. Metrikler: Colab AUROC, mAP, gecikme tablosu

## 10. Demo senaryosu (canlı, ~5 dk)
1. Sağlam lens örneği → **KABUL**, yüksek güven, VLM çağrılmadı (erken çıkış).
2. Çizikli örnek → **RET**, ısı haritası + YOLO maskesi + VLM gerekçesi.
3. Sınır vaka → **İNSAN İNCELEMESİ**, motorlar çelişiyor; kullanıcı tipi değiştirip kaydeder.
4. Geçmiş sekmesi → AI-insan uyumu, CSV (ERP tek giriş), YOLO etiket export (aktif öğrenme).
5. Değerlendiriciye link: kendileri yüklesin.
6. 1 dk mimari + "iki aşamalı sistem" + üretim yol haritası.

## 11. Başlamadan önce onay gerekenler
- [ ] Veri eşlemesi (bölüm 1) uygun mu
- [ ] HF Space adı ve GitHub repo public olsun mu
- [ ] HF token + Gemini anahtarı ortamda (`! export ...` ile ya da HF secret'a elle)
- [ ] "başla"
