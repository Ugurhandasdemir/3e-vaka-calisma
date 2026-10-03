# 3E Elektro Optik — AI Destekli Kalite Kontrol PoC: Teknik Doküman

**Uygulama:** https://qc.ugurhandasdemir.com (yedek: https://huggingface.co/spaces/ugurhandasdemir/3e-vaka-calisma) · **Kod:** https://github.com/Ugurhandasdemir/3e-vaka-calisma

### 1. Problem ve çözüm
Günde 85 ürün elle muayene ediliyor, kayıtlar kâğıt / Excel / ERP arasında tekrar giriliyor, 1.200 kusur görseli etiketsiz ve son 6 ayda 3 kusur müşteriye ulaştı. PoC, muayeneyi **iki aşamalı** bir akışa çeviriyor: **Üretim İçi Muayene** (görsel kusurlar, yapay zekâ) ve **Üretim Sonrası Test** (optik eksen, keskinlik, gevşeme, ölçüm). İki aşama aynı seri numarası ve denetim kaydını paylaşır; ürün ancak ikisi de geçerse **SEVKE HAZIR** olur. Yapay zekâ karar **önerir**, son karar her zaman muayene uzmanındadır.

### 2. Kullanılan yapay zekâ modelleri ve teknolojiler
| Katman | Yöntem | Neden |
|---|---|---|
| Anomali (ana motor) | **PatchCore-WRN50 + AnomalyDINO (DINOv2 ViT-S/14)** ensemble; skorlar "önce birleştir, sonra kalibre et" ile tek bir conformal p-değerine çevrilir | Etiket gerektirmez, yalnız sağlam görsellerle çalışır. İki model farklı hataları yakalar; bu birleştirme bağımlı modellerde de yanlış alarm garantisini korur |
| Kusur tipi | **YOLO11n** (Colab'da, 10 MVTec kategorisinin gerçek kusurları + augmentation, ~1.000 eğitim görseli) | Çizik / kaplama / konektör-montaj sınıfı ve konumu |
| Gerekçe | **Gemini 3.5 Flash-Lite** (yedek 3.8 Flash, 3.1 Flash-Lite); ısı haritası ve kutular ipucu olarak verilir, 3 bağımsız cevabın tutarlılığı ölçülür | İnsanın okuyacağı açıklama; tek başına karar vermez |
| Karar füzyonu | Ağırlıklı skor + oylama → **KABUL / İNSAN İNCELEMESİ / RET**; motorlar çelişirse ya da skor sınırdaysa insana gider; termal modülde otomatik kabul yok | Kaçan kusuru önlemek için belirsizliği insana devreder |
| Etiketleme | Operatör kutu çizer, **MobileSAM** maskeyi çıkarır; YOLO detect/segment formatında dışa aktarılır | 1.200 etiketsiz görselin etiketlenmesi ve aktif öğrenme |
| Son test | **Boresight** (alt-piksel retikül merkezi, görünür/termal kanal arası, titreşim sonrası kayma), **MTF** (ISO 12233 eğik kenar), **tork işareti** (boya çizgisi açı farkı); klasik görüntü işleme | Ölçüm gerektiren kusurlar; açıklanabilir ve izlenebilir |

### 3. Veri yaklaşımı
- Şirket verisi olmadığından **MVTec AD** (CC BY-NC-SA 4.0) temsili veri olarak kullanıldı: metal_nut → optik lens grubu, transistor → termal kamera modülü, cable → gözetleme ünitesi.
- Anomali motoru yalnız **sağlam** görsellerle kurulur (kategori başına 60 referans + 40 kalibrasyon görseli). Bu, etiketsiz başlangıç sorununu çözer.
- Operatörün her kararı ve çizdiği kutu/maske etiketli veriye dönüşür; dışa aktarılan set YOLO'nun yeniden eğitimini besler.
- Ölçüm modülleri, sonucu önceden bilinen **sentetik hedeflerle** doğrulandı.

### 4. Mimari
```
[Görsel + seri no] → Anomali ensemble (WRN50 + DINOv2) → ısı haritası + conformal p
                   → YOLO11n (kusur tipi, kutu) → Gemini (gerekçe, 3 cevap)
                   → Füzyon: KABUL | İNSAN İNCELEMESİ | RET → Uzman kararı (+ kutu / SAM etiketi)
[Son test]         → Boresight (mrad, GEÇTİ/KALDI) · MTF50 · Tork işareti
Kalite kapısı: görsel KABUL + boresight GEÇTİ → SEVKE HAZIR; eksik → BEKLEMEDE; başarısız → RET
Kayıt: SQLite (görsel, model sürümleri, AI önerisi, insan kararı, kullanıcı, zaman) → CSV / YOLO etiket
Altyapı: Gradio + Docker; ev sunucusu (Cloudflare Tunnel, kalıcı disk, 2 GB bellek sınırı) + Hugging Face yedeği
```

### 5. Ölçülen sonuçlar (365 gerçek MVTec test görseli, ayar / test ayrımı)
| Ölçüm | Sonuç |
|---|---|
| Anomali AUROC (test) | WRN50 0,988 → **ensemble 0,995** |
| p ≤ 0,05'te yakalama / yanlış alarm | WRN50 %91,2 / %4,3 → **ensemble %92,9 / %0** |
| Uçtan uca (anomali + YOLO, test) | **0 kaçan kusur, 0 yanlış ret**, %20,2 insan incelemesi |
| YOLO11n mAP50 | 0,814 (uygulama kategorilerinde 0,779) |
| Boresight / MTF50 / tork açısı (sentetik) | maks 0,06 px · maks %1,9 · maks 1,34° (19/19 doğru) |
| Gecikme | ~4–8 sn (VLM dahil, CPU) |

Ayrıntılar: `eval_results/RAPOR.md`, `eval_results/BIRLESTIRME.md`.

### 6. Güvenlik
- API anahtarı yalnız ortam değişkeninde / secret'ta tutulur; kodda ve repoda yoktur. Yüklemeler 10 MB ile sınırlı, RGB'ye çevrilip küçültülür; eşzamanlı iş sınırı vardır.
- Her karar izlenebilir: görsel, model sürümleri, AI önerisi, insan kararı, kullanıcı ve zaman (AS9100 izlenebilirlik mantığı).
- Yapay zekâ karar vermez, önerir; belirsiz durumda "AI önerisini onayla" seçeneği kapanır, uzman seçmek zorundadır.
- PoC'de görseller Google API'ye gider; üretimde veri kurum dışına çıkmamalıdır (bkz. bölüm 8).

### 7. Sınırlılıklar
- Veri temsilidir: MVTec lens değildir; şeffaf / yansıtıcı yüzeylerde başarı düşebilir (MVTec AD 2'de en iyi yöntemler ~%59 AU-PRO).
- YOLO, MVTec test görsellerinin bir kısmıyla eğitildi; uçtan uca sonuçlar dedektör açısından iyimserdir.
- Kalibrasyon seti 40 görsel: p-değeri çözünürlüğü 1/41; %1 yanlış alarm iddiası için sınıf başına ~300 görsel gerekir. İnceleme oranının %20 olmasının ana nedeni budur.
- Ölçüm modülleri sentetik hedeflerle doğrulandı; MTF ve tork sonuçları henüz kalite kapısına kaydedilmiyor. Konektör tutma kuvveti, NETD gibi testler donanım gerektirir.
- VLM'in kendi güven değeri kalibre değildir; tutarlılık ölçülür ama doğru cevabı garanti etmez.

### 8. Üretime geçiş
1. **Veri:** Kontrollü aydınlatma ve fikstürle gerçek görüntü toplama; 1.200 görselin SAM ile ön etiketlenip uzman onayından geçmesi; sınıf başına ≥300 sağlam kalibrasyon görseli.
2. **On-prem:** VLM'in kurum içi küçük bir modelle (ör. Qwen-VL, AD-Copilot / IAD-R1 yaklaşımı) değiştirilmesi; GPU sunucu, rol tabanlı yetkilendirme, değiştirilemez denetim kaydı.
3. **Modeller:** Gerçek veriyle YOLO segmentasyon eğitimi; Dinomaly2 gibi çok sınıflı tek anomali modelinin denenmesi; füzyon ağırlıklarının operatör kararlarından öğrenilmesi.
4. **Son test:** Kolimatör, ısıtılmış hedef ve kenar hedefinin istasyona entegrasyonu; ölçüm belirsizliği ve tolerans tampon bölgesi; akıllı sıkma aletlerinden tork–açı eğrisi verisi.
5. **Doğrulama ve entegrasyon:** Mevcut süreçle paralel gölge pilot, MSA / Gage R&R; ERP/MES'e seri no ve iş emri entegrasyonu; model izleme ve periyodik yeniden eğitim.

---
*Geliştirmede AI kodlama asistanları (Claude, Gemini) kullanıldı; mimari, yöntem seçimleri ve deney tasarımı aday tarafından yapıldı.*
