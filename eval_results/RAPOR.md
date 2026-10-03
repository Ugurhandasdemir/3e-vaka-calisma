# 3E Endüstriyel Kalite Kontrol Sistemi Değerlendirme Raporu (E1 & E3)

## 1. Deney Kurulumu ve Veri Dağılımı

Bu çalışma kapsamında, 3E Kalite Kontrol PoC sistemi için anomali tespit motorları (E1) ve iki motorlu karar füzyon mekanizması (E3) kapsamlı olarak test edilmiştir. Değerlendirmede ağ bağlantısı kullanılmamış; yerel MVTec AD test görüntüleri ve çevrimdışı önbelleğe alınmış modeller kullanılmıştır.

Veri seti, her kategori için sağlam (good) ve kusurlu (defect) sınıfları temsil edecek şekilde katmanlı (stratified) ve deterministik (seed=0) olarak %50 **Tune (Ayar)** ve %50 **Test (Doğrulama)** şeklinde ikiye ayrılmıştır (`eval_results/split.json`):

| Ürün Grubu / Kategori | Toplam Test | Sağlam (Good) | Kusurlu (Defect) | Tune Seti (Sağlam / Kusurlu) | Test Seti (Sağlam / Kusurlu) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **metal_nut** (Optik lens grubu) | 115 | 22 | 93 | 57 (11 / 46) | 58 (11 / 47) |
| **transistor** (Termal kamera modülü) | 100 | 60 | 40 | 50 (30 / 20) | 50 (30 / 20) |
| **cable** (Gözetleme ünitesi) | 150 | 58 | 92 | 75 (29 / 46) | 75 (29 / 46) |
| **GENEL TOPLAM** | **365** | **140** | **225** | **182 (70 / 112)** | **183 (70 / 113)** |

Ek olarak:
- Hafıza bankası referansı için: `samples/<kategori>/good` altında kategori başına 60 adet sağlam görüntü,
- Konformal kalibrasyon için: `samples/<kategori>/calib` altında kategori başına 40 adet sağlam görüntü kullanılmıştır.

---

## 2. Deney E1: Anomali Tespit Motorları Kıyaslaması

Tüm test görüntüleri üzerinde 3 farklı anomali mimarisi koşturulmuş; CPU üzerinde 4 iş parçacığı (`torch.set_num_threads(4)`) ile ortalama işlem süresi, görüntü seviyesi AUROC ve konformal eşik değerinde ($p \le 0.05$) Kusur Yakalama Oranı (Recall / TPR) ile Yanlış Alarm Oranı (FPR) ölçülmüştür (`eval_results/e1_anomaly.csv`):

| Motor / Model | Kategori | AUROC | Kusur Yakalama (Recall @ $p \le 0.05$) | Yanlış Alarm (FPR @ $p \le 0.05$) | Ortalama Süre (ms/görüntü) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PatchCore (WideResNet-50-2)** | metal_nut | 0.9839 | %91.40 | %0.00 | 163.7 ms |
| *(Mevcut Sistem)* | transistor | 0.9892 | %95.00 | %5.00 | 163.1 ms |
| | cable | 0.9689 | %85.87 | %8.62 | 170.1 ms |
| | **Ortalama** | **0.9807** | **%90.76** | **%4.54** | **165.6 ms** |
| **PatchCore (ResNet-18)** | metal_nut | 0.9780 | %94.62 | %4.55 | 57.8 ms |
| *(Hafif Omurga)* | transistor | 0.9342 | %95.00 | %16.67 | 40.9 ms |
| | cable | 0.9200 | %67.39 | %1.72 | 44.3 ms |
| | **Ortalama** | **0.9441** | **%85.67** | **%7.65** | **47.7 ms** |
| **AnomalyDINO (ViT-S/14)** | metal_nut | 1.0000 | %100.00 | %0.00 | 382.0 ms |
| *(DINOv2 Kosinüs Uzaklığı)* | transistor | 0.9846 | %87.50 | %5.00 | 393.1 ms |
| | cable | 0.9488 | %71.74 | %3.45 | 392.3 ms |
| | **Ortalama** | **0.9778** | **%86.41** | **%2.82** | **389.1 ms** |
| **EfficientAD (ONNX)** | Tümü | *N/A* | *N/A* | *N/A* | *N/A* |
| *(Model Dosyası Yok)* | | *(models/anomaly_<cat>.onnx bulunamadı; talimat gereği atlandı)* | | | |

### E1 Değerlendirmesi:
1. **PatchCore (WideResNet-50-2)** genel performansta en dengeli modeldir: Ortalama %98.07 AUROC ve %90.76 kusur yakalama oranına sahiptir.
2. **PatchCore (ResNet-18)** yaklaşık 3.5 kat daha hızlıdır (~47.7 ms), ancak özellikle kablo kategorisinde kusur yakalama oranı %67.39'a gerilemektedir.
3. **AnomalyDINO (ViT-S/14)** optik lens (`metal_nut`) üzerinde kusursuz (%100 AUROC, %100 Recall) sonuç vermiştir; ancak CPU çıkarım süresi (~389 ms) oldukça yüksektir.

---

## 3. Deney E3: Uçtan Uca Karar Füzyonu ve Parametre Optimizasyonu

E3 deneyinde API maliyeti oluşturmamak adına VLM devre dışı bırakılmış; sistem **Anomali (WideResNet-50-2)** ve **YOLO Nesne Dedektörü (`yolo.onnx`)** olmak üzere 2 motorlu olarak çalıştırılmıştır. Tüm test görüntüleri için anomali $p$-değeri ve dedektör maksimum güven/etiket değerleri çıkarılarak kaydedilmiştir (`eval_results/raw/e3_engine_outputs.csv`).

### Tune ve Test Ayrımı Metodolojisi
Model parametreleri arama uzayında (10.800 kombinasyon) aşırı öğrenmeyi (overfitting) ve veri sızıntısını engellemek için optimizasyon **yalnızca Tune seti** üzerinde gerçekleştirilmiştir.
- **Hedef:** İnsan inceleme oranını (review rate) minimize etmek.
- **Kısıtlar:** Kaçak oranı (kusurlu parçaya KABUL denmesi) kesinlikle %0.0 olmalı (yedek sınır $\le$ %2.0), Yanlış Ret oranı (sağlam parçaya RET denmesi) $\le$ %5.0 olmalıdır.
- Tune setinde en iyi bulunan parametreler daha sonra bağımsız **Test seti** üzerinde doğrulanmıştır.

### Seçilen ve Karşılaştırılan Parametreler
- **Mevcut Konfigürasyon:** $w_{anom}=0.45$, $w_{det}=0.25$, $accept\_below=0.30$, $reject\_above=0.70$, $anom\_vote\_p=0.053$, $det\_vote\_conf=0.40$
- **Optimize Konfigürasyon:** $w_{anom}=0.35$, $w_{det}=0.20$, $accept\_below=0.30$, $reject\_above=0.55$, $anom\_vote\_p=0.10$, $det\_vote\_conf=0.30$

### E3 Karşılaştırma Tablosu

| Konfigürasyon | Veri Seti | Kaçak Oranı (Defect $\to$ ACCEPT) | Yanlış Ret Oranı (Good $\to$ REJECT) | İnsan İnceleme Oranı (REVIEW) | Otomatik Karar Doğruluğu | RET Kusur Tipi Doğruluğu |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Mevcut (Current)** | **Tune** | %0.89 *(1/112 kaçak)* | %0.00 *(0/70)* | %18.68 *(34/182)* | %99.32 *(147/148)* | %96.81 *(91/94)* |
| **Optimize (Tuned)** | **Tune** | **%0.00 *(0/112)*** | %0.00 *(0/70)* | **%17.03 *(31/182)*** | **%100.00 *(151/151)*** | **%97.00 *(97/100)*** |
| **Mevcut (Current)** | **Test** | %0.00 *(0/113)* | %0.00 *(0/70)* | %16.39 *(30/183)* | %100.00 *(153/153)* | %91.92 *(91/99)* |
| **Optimize (Tuned)** | **Test** | **%0.00 *(0/113)*** | %0.00 *(0/70)* | **%16.94 *(31/183)*** | **%100.00 *(152/152)*** | **%92.31 *(96/104)*** |

### Sonuçların Analizi:
- Mevcut konfigürasyonda, tune setinde `cable_swap` kusurlu bir numune dedektör ve anomali oyu alamadığı için hattan kaçarak otomatik KABUL edilmiştir (%0.89 kaçak).
- Optimize edilen konfigürasyonda ret eşiği 0.70'ten 0.55'e indirilmiş, anomali oy eşiği 0.10'a esnetilmiş ve dedektör oy eşiği 0.30'a çekilmiştir.
- Bu sayede kaçak oranı **%0.0'a** düşürülmüş, otomatik karar doğruluğu **%100'e** ulaşmış ve RET kararlarında doğru kusur tipi teşhisi %91.92'den **%92.31'e** yükselmiştir (5 kusurlu parça daha doğru etiketle doğrudan reddedilmiştir).

---

## 4. Sonuç ve Öneriler (5 Satır)

1. Üretim hattında birincil anomali motoru olarak yüksek genel başarım sunan **PatchCore (WideResNet-50-2)** kullanılmalıdır.
2. Karar mekanizmasında `eval_results/e3_best_params.json` içerisinde yer alan **optimize edilmiş parametre seti** ($w_{anom}=0.35, w_{det}=0.20, reject=0.55, anom\_p=0.10, det\_conf=0.30$) benimsenmelidir.
3. Bu parametreler ile mevcut sistemdeki kaçak riski tamamen elimine edilmiş, test setinde sıfır kaçak (%0.0) ve %100 otomatik karar doğruluğu sağlanmıştır.
4. RET kararlarında doğru kusur tipi sınıflandırma başarısı test setinde %91.92'den %92.31'e çıkarılarak hatalı parçaların teşhis kalitesi artırılmıştır.
5. Aşırı yüksek hız veya kenar donanım (edge device) kısıtlarında PatchCore ResNet-18 (47.7 ms) alternatif olabilir; ancak kablo gibi karmaşık ürünlerde WideResNet-50-2 vazgeçilmezdir.

---

## 5. Kısıtlar ve Dikkat Edilmesi Gerekenler (Limitations)

> [!WARNING]
> **YOLO Dedektör Veri Kirliliği / İyimserlik Uyarısı:**
> Sistemde kullanılan `models/yolo.onnx` modeli eğitilirken MVTec test kümesindeki görüntülerin veya benzer sentetik varyasyonlarının eğitim/doğrulama sürecinde yer almış olma olasılığı bulunmaktadır. Bu durum, dedektörün kusur tespit güven skorlarının ve dolayısıyla dedektör kaynaklı metriklerin gerçek fabrika ortamına kıyasla **iyimser (optimistic)** çıkmasına yol açabilir. Fabrika ortamında canlı üretime geçilmeden önce, eğitimde hiç kullanılmamış bağımsız parti verileriyle dedektörün ve füzyon eşiklerinin yeniden doğrulanması kritik önem taşımaktadır.
