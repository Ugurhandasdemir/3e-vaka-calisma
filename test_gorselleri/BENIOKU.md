# Test Görselleri

Bu klasör değerlendirme için hazırlanmış örnek girdileri içerir. Görseller uygulamaya yüklenerek denenebilir.

- `gorsel_muayene/`: Üretim İçi Muayene sekmesi için. Üç alt klasör, üç ürün grubuna karşılık gelir. Her klasörde 2 sağlam ve 3 kusurlu görsel vardır.
- `son_test/`: Üretim Sonrası Test sekmesi için (boresight, MTF, tork işareti).

## Kullanım

1. Üretim İçi Muayene sekmesinde ürün grubunu klasör adına uygun seçin (aşağıdaki tablo), görseli yükleyin, Parçayı Analiz Et düğmesine basın.
2. Üretim Sonrası Test sekmesinde ilgili alt sekmeye (Boresight, MTF, Tork İşareti) geçip görseli yükleyin. Boresight için kanal ve optik parametreleri aşağıda verilmiştir.

## Görsel muayene: beklenen sonuçlar

Aşağıdaki kararlar `pipeline.run` ile yerelde, Gemini anahtarı olmadan (VLM olmadan) üretilmiştir. VLM ile sonuç biraz değişebilir. Görseller MVTec AD test bölümünden alınmıştır; modelin ayar/kalibrasyon aşamasında kullanılmayan örneklerdir. Anomali motoru: PatchCore + DINOv2 topluluğu.


### `gorsel_muayene/optik_lens/` — Optik lens grubu (metal_nut)

| Dosya | Gerçek durum | Beklenen karar | Beklenen kusur tipi |
| --- | --- | --- | --- |
| `saglam_1.png` | Sağlam | KABUL önerisi | - |
| `saglam_2.png` | Sağlam | İNSAN İNCELEMESİ | Diğer |
| `kusurlu_scratch_1.png` | Kusurlu (Yüzey çiziği) | RET önerisi | Yüzey çiziği |
| `kusurlu_bent_1.png` | Kusurlu (Konektör/montaj kusuru) | RET önerisi | Konektör/montaj kusuru |
| `kusurlu_flip_1.png` | Kusurlu (Konektör/montaj kusuru) | RET önerisi | Konektör/montaj kusuru |

### `gorsel_muayene/termal_modul/` — Termal kamera modülü (transistor, yüksek risk)

| Dosya | Gerçek durum | Beklenen karar | Beklenen kusur tipi |
| --- | --- | --- | --- |
| `saglam_1.png` | Sağlam | KABUL önerisi | - |
| `saglam_2.png` | Sağlam | KABUL önerisi | - |
| `kusurlu_damaged_case_1.png` | Kusurlu (Yüzey çiziği) | RET önerisi | Yüzey çiziği |
| `kusurlu_misplaced_1.png` | Kusurlu (Konektör/montaj kusuru) | RET önerisi | Konektör/montaj kusuru |
| `kusurlu_bent_lead_1.png` | Kusurlu (Konektör/montaj kusuru) | İNSAN İNCELEMESİ | Diğer |

### `gorsel_muayene/gozetleme_unitesi/` — Gözetleme ünitesi (cable)

| Dosya | Gerçek durum | Beklenen karar | Beklenen kusur tipi |
| --- | --- | --- | --- |
| `saglam_1.png` | Sağlam | KABUL önerisi | - |
| `saglam_2.png` | Sağlam | KABUL önerisi | - |
| `kusurlu_missing_wire_1.png` | Kusurlu (Konektör/montaj kusuru) | RET önerisi | Konektör/montaj kusuru |
| `kusurlu_poke_insulation_1.png` | Kusurlu (Kaplama kusuru) | RET önerisi | Kaplama kusuru |
| `kusurlu_cut_outer_insulation_1.png` | Kusurlu (Kaplama kusuru) | RET önerisi | Kaplama kusuru |

Not: sağlam örneklerden biri (`optik_lens/saglam_2.png`) sınırda kalıp İNSAN İNCELEMESİ üretir; bu, sistemin şüpheli durumda operatöre bıraktığını göstermek için bilerek korunmuştur.

## Son test: beklenen sonuçlar

### `son_test/boresight/` (tolerans 0,5 mrad)

| Dosya | Kanal | Piksel boyutu / odak | Beklenen sonuç |
| --- | --- | --- | --- |
| `vis_offset_3_2_az.png` | Görünür | 3,45 µm / 50 mm | PASS (0,22 mrad) |
| `vis_offset_15_0_el.png` | Görünür | 3,45 µm / 50 mm | FAIL (1,03 mrad) |
| `thm_offset_subpix_pass.png` | Termal | 12,0 µm / 25 mm | PASS (0,37 mrad) |

### `son_test/mtf/`

Eğik kenar (ISO 12233) hedefleri. Görünür kanal 3,45 µm, termal 12,0 µm piksel boyutu ile girilir.

| Dosya | Kanal | Beklenen MTF50 |
| --- | --- | --- |
| `vis_sigma_1.0.png` | Görünür | yaklaşık 0,187 cy/px (54,2 lp/mm), keskin |
| `thm_sigma_2.5.png` | Termal | yaklaşık 0,076 cy/px (6,3 lp/mm), bulanık |

### `son_test/tork_isareti/` (boya rengi: kırmızı, tolerans 5°)

| Dosya | Durum | Beklenen karar |
| --- | --- | --- |
| `kirmizi_rot00_gecer.png` | Boya çizgisi kesintisiz | GEÇTİ (açı farkı 0,0°) |
| `kirmizi_rot30_kalir.png` | Vida 30° dönmüş | KALDI (açı farkı 29,7°) |
| `kirmizi_boya_eksik.png` | Vida kafasında boya yok | KALDI (işaret eksik) |

## Lisans ve atıf

Görsel muayene görselleri MVTec AD veri kümesinden alınmış olup lisansı CC BY-NC-SA 4.0'dır (yalnızca araştırma, eğitim ve PoC amaçlı). Kaynak: Bergmann ve ark., CVPR 2019. Son test görselleri bu proje için sentetik üretilmiştir.

> **Not:** `termal_modul/kusurlu_bent_lead_1.png` kusurlu olduğu halde otomatik RET yerine **İNSAN İNCELEMESİ**'ne düşer. Bu bir kaçak değildir: termal modül yüksek riskli grup olduğu için sistem emin olmadığı parçayı uzmana gönderir, kusur müşteriye ulaşmaz.
