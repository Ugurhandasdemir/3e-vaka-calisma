# İki Bağımlı Anomali Tespit Modelinin Birleştirilmesi ve Konformal Güvence Raporu (E1b)

Bu rapor, 3E Endüstriyel Kalite Kontrol Sistemi kapsamında kullanılan iki bağımlı ve denetimsiz anomali tespit motorunun (**PatchCore-WRN50** ve **AnomalyDINO ViT-S/14**) skor ve p-değerlerinin birleştirilmesini; küçük kalibrasyon kümesi ($n=40$ sağlam görüntü/kategori) altında geçerli tip I hata (yanlış alarm, FPR) güvencesinin korunmasını teorik ve deneysel olarak ele almaktadır.

---

## BÖLÜM 1: Literatür İncelemesi (Part 1 — Literature Research)

İki modelin de aynı görüntü üzerinde çalışması nedeniyle ürettikleri skorlar ve p-değerleri **istatistiksel olarak bağımlıdır (pozitif korelasyonlu)**. Bağımlılık yapısı bilinmediğinde veya karmaşık olduğunda, yanlış alarm garantisinin ($\text{FPR} \le \alpha$) korunabilmesi için seçilen birleştirme metodolojisinin bu bağımlılığa dayanıklı olması zorunludur.

### 1.1 İncelenen Yaklaşımlar ve Teorik Temeller

1. **Skor Normalizasyonu ve Sezgisel Kümeleme (Heuristic Aggregation):**
   - *z-score* (kalibrasyon ortalaması ve standart sapması ile) ve *min-max* normalizasyonu, farklı dinamik aralıklara sahip skorları ortak bir ölçeğe taşır.
   - **Kriegel et al. (2011)**, farklı anomali faktörlerini standartlaştırıp $[0, 1]$ aralığına dönüştürerek "anomali olasılığı" olarak yorumlamayı önermiştir.
   - **Aggarwal & Sathe (2015/2017)**, denetimsiz aykırı değer topluluklarında (outlier ensembles) varyans azaltma mekanizmalarını incelemiş, ortalama, maksimum ve sıra ortalaması (rank average) gibi yöntemleri analiz etmiştir.
   - **LSCP (Zhao et al., 2019)** ve **SUOD (Zhao et al., 2021)**, yüksek boyutlu ve heterojen dedektör havuzlarında yerel komşuluk veya rastgele projeksiyon ile model seçimi ve ölçeklenebilirlik sağlar; ancak bu yaklaşımlar dağılımdan bağımsız sonlu örneklem yanlış alarm garantisi vermez.

2. **Keyfi Bağımlılık Altında Geçerli p-Değeri Birleştirme Yöntemleri:**
   - **Bonferroni Düzeltmesi ($\min(p_1, p_2) \times 2$):** Boole-Bonferroni eşitsizliğine (union bound) dayanır. Testler arasındaki bağımlılık yapısı ne olursa olsun geçerlidir; ancak pozitif korelasyon altında aşırı tutucudur (conservative).
   - **Vovk & Wang (2020) — Aritmetik Ortalama ($2 \bar{p} = p_1 + p_2$):** Keyfi bağımlılık altında p-değerlerinin ortalamasının 2 katının daima geçerli bir p-değeri olduğunu ispatlamışlardır.
   - **Cauchy Birleştirme Testi / ACAT (Liu & Xie, 2020):** Bireysel p-değerlerini Cauchy dağılımına dönüştürerek toplar. Cauchy dağılımının ağır kuyruklu toplam kararlılığı sayesinde, keyfi bağımlılık yapıları altında asimptotik olarak geçerlidir ve özellikle seyrek güçlü sinyallere karşı çok hassastır.
   - **Harmonik Ortalama p-Değeri (Wilson, 2019):** Bağımlı testler için önerilmiş olsa da, Goeman & Rosenblatt (2019) ve Vovk & Wang (2020) ham harmonik ortalamanın sonlu örneklemde keyfi bağımlılık altında anti-konservatif (geçersiz) kalabileceğini, geçerlilik için $e \ln K$ düzeltme çarpanı gerektiğini göstermiştir.

3. **Bağımsızlık Varsayan Klasik p-Değeri Birleştirme Yöntemleri:**
   - **Fisher (1925/1932):** $-2 \sum \ln(p_i) \sim \chi^2(2K)$ istatistiğini kullanır. Testlerin tam bağımsız olduğunu varsayar. Pozitif bağımlı modellerde p-değerlerini aşırı küçük hesaplar ve yanlış alarm oranını şişirir (geçersizdir).
   - **Stouffer et al. (1949):** Normal z-skor dönüşümü $Z = \frac{1}{\sqrt{K}} \sum \Phi^{-1}(1 - p_i)$ kullanır; Fisher gibi tam bağımsızlık varsayar.

4. **Konformal "Önce Skorları Birleştir, Sonra Tek Seferde Kalibre Et" (Combine-then-Calibrate):**
   - **Vovk et al. (2005)** ve **Angelopoulos & Bates (2021)** tarafından ortaya konan İndüktif Konformal Tahmin (Split Conformal Prediction) teorisine göre; herhangi bir tekil skaler uyumsuzluk skoru (nonconformity score) $S(x)$ seçilirse, bu skorun nasıl oluşturulduğuna (örneğin iki modelin z-skorlarının ağırlıklı ortalaması) bakılmaksızın, kalibrasyon ve test örnekleri aynı dağılımdan geldikçe (değiştirilebilirlik / exchangeability altında):
     $$p(x) = \frac{1 + \sum_{i=1}^n \mathbf{1}\{S(X_i^{\text{cal}}) \ge S(x)\}}{n+1}$$
     formülü **keyfi model bağımlılığı altında sonlu örneklemde ($n=40$) KESİN ve GEÇERLİ bir yanlış alarm garantisi verir ($P(p(X) \le \alpha) \le \alpha$)**. Modeller arasındaki korelasyonu kalibrasyon kümesi doğal olarak bünyesinde barındırır.

5. **E-Değerleri Birleştirme (Vovk & Wang, 2021):**
   - $H_0$ altında beklenen değeri $\le 1$ olan e-değerleri, keyfi bağımlılık altında doğrudan aritmetik ortalama alınarak birleştirilebilir. Ancak p-değerinden e-değerine dönüşüm (kalibratör) küçük örneklemde ek güç kaybına yol açar.

6. **Öğrenilmiş Yığınlama / Learned Stacking (Wolpert, 1992 - Lojistik Regresyon):**
   - Modellerin skorlarını girdi alarak kusur etiketini tahmin eden bir meta-model eğitilmesidir. Ancak denetimsiz anomali tespitinde eğitim anında kusurlu veri bulunmaz; çok küçük doğrulama setlerinde ise aşırı öğrenme (overfitting) riski büyüktür ve sonlu örneklemde tip I hata kontrolü sağlamaz.

---

### 1.2 Literatür Doğrulama Tablosu

| Yöntem | İlgili Temel Makale | Yıl / Venue | Doğrulanmış URL | Keyfi Bağımlılıkta Geçerli mi? | Not ve Değerlendirme |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Aykırı Değer Skorlarını Birleştirme (z-score / min-max)** | Kriegel, Kröger, Schubert, Zimek — *Interpreting and Unifying Outlier Scores* | 2011 (SDM) | [10.1137/1.9781611972818.2](https://doi.org/10.1137/1.9781611972818.2) | **HAYIR** | Skorları $[0,1]$ olasılık uzayına normalize eder; istatistiksel hata garantisi vermez. |
| **Aykırı Değer Toplulukları Kuramı** | Aggarwal, Sathe — *Theoretical Foundations and Algorithms for Outlier Ensembles* | 2015 (SIGKDD Expl.) | [10.1145/2830544.2830549](https://doi.org/10.1145/2830544.2830549) | **HAYIR** | Model korelasyonunu ve varyans düşümünü irdeler; sezgisel birleştirmedir, p-değeri garantisi yoktur. |
| **Yerel Seçici Birleştirme (LSCP)** | Zhao, Nasrullah, Hryniewicki, Li et al. — *LSCP: Locally Selective Combination in Parallel Outlier Ensembles* | 2019 (SDM) | [10.1137/1.9781611975673.66](https://doi.org/10.1137/1.9781611975673.66) | **HAYIR** | Test noktası komşuluğunda en iyi dedektörü seçer; tip I hata kontrolü sunmaz. |
| **Büyük Ölçekli Heterojen Topluluk (SUOD)** | Zhao, Hu, Cheng et al. — *SUOD: Accelerating Large-Scale Unsupervised Heterogeneous Outlier Detection* | 2021 (MLSys) | [MLSys 2021 Paper](https://proceedings.mlsys.org/paper/2021/file/e4da3b7fbbce2345d7772b0674a318d5-Paper.pdf) | **HAYIR** | Rastgele projeksiyonla hesaplama hızlandırma odaklıdır; p-değeri garantisi sağlamaz. |
| **Bonferroni Düzeltmesi ($\min(p_1, p_2) \times 2$)** | Dunn — *Multiple Comparisons Among Means* (Boole-Bonferroni Eşitsizliği) | 1961 (JASA) | [10.1080/01621459.1961.10482090](https://doi.org/10.1080/01621459.1961.10482090) | **EVET** | Keyfi bağımlılık altında daima geçerlidir; ancak pozitif bağımlılıkta aşırı tutucudur ($n=40$ için $p \le 0.025$ şartı arar). |
| **Ortalama ile p-Değeri Birleştirme ($p_1+p_2$)** | Vovk, Wang — *Combining p-values via averaging* | 2020 (Biometrika) | [10.1093/biomet/asaa027](https://doi.org/10.1093/biomet/asaa027) | **EVET** | $2 \bar{p} = p_1 + p_2$ keyfi bağımlılıkta kesin geçerlidir; $n=40$ durumunda her iki modelin de en küçük p vermesini şart koşar, güç çöker. |
| **Cauchy Birleştirme Testi (ACAT)** | Liu, Xie — *Cauchy combination test: a powerful test with analytic p-value calculation under arbitrary dependency structures* | 2020 (JASA) | [10.1080/01621459.2018.1554485](https://doi.org/10.1080/01621459.2018.1554485) | **EVET (Asimptotik)** | Bivariate Cauchy kuyruk özelliğiyle keyfi bağımlılıkta geçerlidir; uç değerlere karşı çok etkilidir. |
| **Harmonik Ortalama p-Değeri (HMP)** | Wilson — *The harmonic mean p-value for combining dependent tests* | 2019 (PNAS) | [10.1073/pnas.1814093116](https://doi.org/10.1073/pnas.1814093116) | **KISMEN / ŞARTLI** | Wilson bağımlılıkta geçerli olduğunu öne sürse de; Vovk & Wang (2020) sonlu örneklemde $e \ln K$ sabiti olmadan anti-konservatif kalabileceğini kanıtlamıştır. |
| **Fisher Birleştirme Testi** | Fisher — *Statistical Methods for Research Workers* | 1925 (Oliver & Boyd) | [hdl.handle.net/2440/15227](https://hdl.handle.net/2440/15227) | **HAYIR** | **Testlerin kesin bağımsız olduğunu varsayar.** Pozitif bağımlılıkta tip I hatayı şişirir (aşırı iyimserdir). |
| **Stouffer Z-Skor Testi** | Stouffer et al. — *The American Soldier: Adjustment During Army Life* | 1949 (Princeton Univ.) | [psycnet.apa.org/record/1950-00782-000](https://psycnet.apa.org/record/1950-00782-000) | **HAYIR** | Bağımsız standart normallik varsayar; bağımlılık altında hatalıdır. |
| **Konformal "Önce Birleştir, Sonra Kalibre Et"** | Vovk, Gammerman, Shafer (2005) / Angelopoulos, Bates (2021) | 2005 (Springer) / 2021 (arXiv) | [10.1007/b106715](https://doi.org/10.1007/b106715) / [arXiv:2107.07511](https://arxiv.org/abs/2107.07511) | **EVET (Tam/Kesin)** | **Sonlu örneklemde ($n=40$) tam ve kesin geçerlidir.** İki model arasındaki bağımlılık yapısından tamamen bağımsızdır; çünkü kalibrasyon birleşik skor üzerinde tek seferde yapılır. |
| **E-Değerleri Birleştirme** | Vovk, Wang — *E-values: Calibration, combination and applications* | 2021 (Ann. Statist.) | [10.1214/20-AOS2020](https://doi.org/10.1214/20-AOS2020) | **EVET** | E-değerlerinin aritmetik ortalaması keyfi bağımlılıkta doğrudan geçerlidir. Ancak p $\to$ e dönüşümü güç kaybı yaratır. |
| **Öğrenilmiş Yığınlama (Learned Stacking)** | Wolpert — *Stacked Generalization* | 1992 (Neural Netw.) | [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1) | **HAYIR** | Denetimli etiket gerektirir; küçük veri setinde aşırı öğrenir ve doğrudan tip I hata güvencesi sunmaz. |

---

## BÖLÜM 2: Deneysel Sonuçlar (Part 2 — Benchmark Results)

Deneyde, `eval_results/raw/e1_scores.csv` içerisindeki **PatchCore (wide_resnet50_2)** ve **AnomalyDINO (anomaly_dino_vits14)** modellerine ait kayıtlı test skorları ve p-değerleri ile `models/` ve `eval_results/cache/` altındaki $n=40$ kalibrasyon verileri kullanılmıştır. Tüm modeller `eval/combine.py` betiği ile koşturulmuş ve `eval_results/e1b_combine.csv` dosyasına kaydedilmiştir.

### 2.1 Doğrulama Seti (Split = "TEST") Sonuçları

| Yöntem | Kategori | AUROC | Kusur Yakalama (Recall @ $p \le 0.05$) | Yanlış Alarm (FPR @ $p \le 0.05$) | Kusur Yakalama (Recall @ $p \le 0.10$) | Yanlış Alarm (FPR @ $p \le 0.10$) | Max Yakalama (FPR $\le 5\%$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **(a) WRN50 tek başına** | metal_nut | 1.0000 | %97.87 | %0.00 | %97.87 | %9.09 | %100.00 |
| | transistor | 0.9867 | %90.00 | %3.33 | %95.00 | %6.67 | %95.00 |
| | cable | 0.9783 | %84.78 | %6.90 | %89.13 | %6.90 | %80.43 |
| | **ORTALAMA** | **0.9883** | **%90.88** | **%3.41** | **%94.71** | **%7.55** | **%91.81** |
| **(b) DINO tek başına** | metal_nut | 1.0000 | %100.00 | %0.00 | %100.00 | %9.09 | %100.00 |
| | transistor | 0.9800 | %85.00 | %6.67 | %95.00 | %6.67 | %75.00 |
| | cable | 0.9640 | %73.91 | %3.45 | %82.61 | %3.45 | %86.96 |
| | **ORTALAMA** | **0.9813** | **%86.30** | **%3.37** | **%92.54** | **%6.40** | **%87.32** |
| **(c) z-score mean (parametrik p)** | metal_nut | 1.0000 | %100.00 | %0.00 | %100.00 | %0.00 | %100.00 |
| | transistor | 0.9983 | %100.00 | %3.33 | %100.00 | %10.00 | %100.00 |
| | cable | 0.9918 | %84.78 | %0.00 | %84.78 | %3.45 | %95.65 |
| | **ORTALAMA** | **0.9967** | **%94.93** | **%1.11** | **%94.93** | **%4.48** | **%98.55** |
| **(d) min-max mean** | metal_nut | 1.0000 | %100.00 | %0.00 | %100.00 | %0.00 | %100.00 |
| | transistor | 0.9983 | %75.00 | %0.00 | %85.00 | %0.00 | %100.00 |
| | cable | 0.9918 | %69.57 | %0.00 | %71.74 | %0.00 | %95.65 |
| | **ORTALAMA** | **0.9967** | **%81.52** | **%0.00** | **%85.58** | **%0.00** | **%98.55** |
| **(e) max of z-scores** | metal_nut | 1.0000 | %100.00 | %0.00 | %100.00 | %9.09 | %100.00 |
| *(Yanlış Alarm İhlali!)* | transistor | 0.9967 | %100.00 | %10.00 | %100.00 | %16.67 | %95.00 |
| | cable | 0.9768 | %86.96 | %10.34 | %95.65 | %10.34 | %86.96 |
| | **ORTALAMA** | **0.9911** | **%95.65** | **%6.78** | **%98.55** | **%12.03** | **%93.99** |
| **(f) combine-then-calibrate** | metal_nut | 1.0000 | %100.00 | %0.00 | %100.00 | %9.09 | %100.00 |
| *(ÖNERİLEN YÖNTEM)* | transistor | 0.9983 | %95.00 | %0.00 | %100.00 | %6.67 | %100.00 |
| | cable | 0.9918 | %84.78 | %0.00 | %89.13 | %3.45 | %95.65 |
| | **ORTALAMA** | **0.9967** | **%93.26** | **%0.00** | **%96.38** | **%6.40** | **%98.55** |
| **(g) Bonferroni min-p*2** | metal_nut | 1.0000 | %100.00 | %0.00 | %100.00 | %0.00 | %100.00 |
| | transistor | 0.9950 | %90.00 | %0.00 | %100.00 | %10.00 | %90.00 |
| | cable | 0.9756 | %78.26 | %0.00 | %91.30 | %10.34 | %78.26 |
| | **ORTALAMA** | **0.9902** | **%89.42** | **%0.00** | **%97.10** | **%6.78** | **%89.42** |
| **(h) Vovk-Wang 2*mean p** | metal_nut | 1.0000 | %93.62 | %0.00 | %97.87 | %0.00 | %100.00 |
| *(n=40'ta Güç Çöküşü!)* | transistor | 0.9983 | %35.00 | %0.00 | %85.00 | %0.00 | %100.00 |
| | cable | 0.9936 | %52.17 | %0.00 | %67.39 | %0.00 | %95.65 |
| | **ORTALAMA** | **0.9973** | **%60.26** | **%0.00** | **%83.42** | **%0.00** | **%98.55** |
| **(i) Cauchy ACAT** | metal_nut | 1.0000 | %100.00 | %0.00 | %100.00 | %0.00 | %100.00 |
| | transistor | 1.0000 | %95.00 | %0.00 | %100.00 | %6.67 | %100.00 |
| | cable | 0.9891 | %82.61 | %0.00 | %89.13 | %6.90 | %89.13 |
| | **ORTALAMA** | **0.9964** | **%92.54** | **%0.00** | **%96.38** | **%4.52** | **%96.38** |
| **(j) Fisher (bağımsızlık varsayan)**| metal_nut | 1.0000 | %100.00 | %0.00 | %100.00 | %9.09 | %100.00 |
| | transistor | 1.0000 | %100.00 | %3.33 | %100.00 | %6.67 | %100.00 |
| | cable | 0.9899 | %86.96 | %0.00 | %91.30 | %3.45 | %91.30 |
| | **ORTALAMA** | **0.9966** | **%95.65** | **%1.11** | **%97.10** | **%6.40** | **%97.10** |
| **(k) Rank average (yalnızca skor)** | metal_nut | 1.0000 | *N/A* | *N/A* | *N/A* | *N/A* | %100.00 |
| | transistor | 1.0000 | *N/A* | *N/A* | *N/A* | *N/A* | %100.00 |
| | cable | 0.9914 | *N/A* | *N/A* | *N/A* | *N/A* | %95.65 |
| | **ORTALAMA** | **0.9971** | *N/A* | *N/A* | *N/A* | *N/A* | **%98.55** |

---

### 2.2 Tüm Veri (Split = "ALL", 365 Görüntü) Makro Özet Tablosu

| Sıra | Yöntem | Ortalama AUROC | Kusur Yakalama (Recall @ $p \le 0.05$) | Yanlış Alarm (FPR @ $p \le 0.05$) | Kusur Yakalama (Recall @ $p \le 0.10$) | Yanlış Alarm (FPR @ $p \le 0.10$) | Max Yakalama (FPR $\le 5\%$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **(f) combine-then-calibrate** | **0.9936** | **%94.09** | **%0.00** | **%96.38** | **%5.44** | **%96.38** |
| 2 | **(i) Cauchy ACAT** | 0.9935 | %93.37 | %0.00 | %96.38 | %5.10 | %96.38 |
| 3 | **(j) Fisher** *(bağımsızlık varsayar)* | 0.9935 | %94.94 | %1.11 | %96.74 | %6.04 | %96.74 |
| 4 | **(c) z-score mean** *(parametrik normal p)* | 0.9936 | %94.93 | %0.56 | %95.65 | %4.89 | %96.38 |
| 5 | **(a) WRN50 tek başına** | 0.9806 | %90.76 | %4.54 | %93.76 | %7.17 | %88.57 |
| 6 | **(g) Bonferroni min-p*2** | 0.9873 | %89.17 | %0.00 | %97.10 | %7.36 | %89.17 |
| 7 | **(b) DINO tek başına** | 0.9778 | %86.41 | %2.82 | %93.01 | %5.44 | %90.51 |
| 8 | **(k) Rank average** *(p-değeri yok)* | 0.9929 | *N/A* | *N/A* | *N/A* | *N/A* | %97.10 |
| 9 | **(d) min-max mean** | 0.9936 | %78.78 | %0.00 | %83.30 | %0.00 | %96.38 |
| 10 | **(e) max of z-scores** *(FPR ihlali)* | 0.9889 | %96.01 | **%7.36** | %98.19 | %13.57 | %94.09 |
| 11 | **(h) Vovk-Wang 2*mean p** | 0.9917 | %55.45 | %0.00 | %81.62 | %0.00 | %96.38 |

---

## BÖLÜM 3: Yöntem Sıralaması (En İyiden En Kötüye)

Bu sıralama, hem **istatistiksel geçerlilik (keyfi bağımlılık altında garanti)** hem de **endüstriyel performans (AUROC, Recall, FPR)** kriterleri birlikte gözetilerek yapılmıştır:

1. 🥇 **(f) combine-then-calibrate (Önce Birleştir, Sonra Kalibre Et):**
   - *Neden En İyi:* İki model arasındaki bağımlılık yapısından tamamen bağımsızdır; sonlu örneklemde ($n=40$) geçerli konformal yanlış alarm garantisi ($\text{FPR} \le 5\%$) sunar. Test setinde %99.67 AUROC ve sıfır yanlış alarm ile %93.26 yakalama elde etmiştir. Tekil modellerin her ikisinden de belirgin biçimde üstündür.
2. 🥈 **(i) Cauchy Birleştirme Testi (ACAT):**
   - *Değerlendirme:* Bireysel p-değerleri üzerinden keyfi bağımlılık altında çalışan en güçlü yöntemdir. Ağır kuyruklu dönüşüm sayesinde p-değerlerini dengeler. Test setinde %99.64 AUROC ve %92.54 kusur yakalama sağlamış, yanlış alarmı %0.00'da tutmuştur.
3. 🥉 **(j) Fisher Birleştirme:**
   - *Değerlendirme:* Ampirik olarak test setinde %95.65 yakalama ve %99.66 AUROC ile çok yüksek performans gösterse de, teorik olarak **bağımsızlık varsayımına dayanır**. Pozitif bağımlı modellerde tip I hatayı şişirme riski taşır (transistor'da %3.33 FPR).
4. **(c) z-Score Mean (Parametrik Normal p):**
   - *Değerlendirme:* Skor seviyesinde çok güçlüdür (AUROC 0.9967); ancak Gaussian varsayımı ampirik anomali dağılımlarına uymadığı için konformal güvenceden yoksundur.
5. **(g) Bonferroni min-p*2:**
   - *Değerlendirme:* Bağımlılık altında %100 geçerlidir; fakat $n=40$ durumunda en küçük ikinci p-değerini ($2/41 \approx 0.0488$) ikiyle çarparak $0.0976$'ya fırlattığı için test setinde yakalama oranı %89.42'ye geriler.
6. **(a) WRN50 Tek Başına:**
   - *Değerlendirme:* Mevcut temel sistem (Test AUROC: 0.9883, Recall: %90.88, FPR: %3.41). Kablo kategorisinde FPR %6.90'a çıkarak %5 sınırını aşmaktadır.
7. **(b) DINO Tek Başına:**
   - *Değerlendirme:* Metal nut üzerinde kusursuz olsa da transistor ve kabloda geride kalmaktadır (Test AUROC: 0.9813, Recall: %86.30).
8. **(k) Rank Average:**
   - *Değerlendirme:* Saf sıralama için harika bir AUROC (%99.71) üretir; ancak operasyonel bir eşikleme için p-değeri üretemez.
9. **(d) min-max Mean:**
   - *Değerlendirme:* Uç değerlere aşırı hassastır; kalibrasyon min-max aralığı dışına çıkan test verilerinde yakalama oranı %81.52'ye kadar düşer.
10. **(h) Vovk-Wang 2*mean p ($p_1+p_2$):**
    - *Değerlendirme:* Teorik olarak keyfi bağımlılıkta geçerli olsa da, $n=40$ durumunda her iki modelin de aynı anda $p=1/41$ vermesini zorunlu kılar ($1/41 + 2/41 = 3/41 > 0.05$). Bu durum kusur yakalama oranını %60.26'ya düşürerek yöntemi pratikte kullanışsız hale getirir.
11. ❌ **(e) max of z-scores:**
    - *Değerlendirme:* **En tehlikeli yöntemdir.** İki z-skorunun maksimumunu almak, yanlış alarm oranını doğrudan katlar. Test setinde ortalama FPR **%6.78**, kabloda ise **%10.34** olarak ölçülmüş; %5'lik kalite kontrol yanlış alarm şartı ağır biçimde ihlal edilmiştir.

---

## BÖLÜM 4: Kesin Öneri ve Uygulama Formülü

### 4.1 Öneri (One Method)
3E PoC sistemi için kesin önerimiz: **"Combine-then-Calibrate" (Önce Skorları Birleştir, Sonra Tek Seferde Kalibre Et)** yöntemidir.

### 4.2 Kesin Matematiksel Formül

1. **Adım 1 — Kalibrasyon İstatistiklerinin Hesaplanması:**
   Her ürün kategorisi ($c$) için elde bulunan $n=40$ adet sağlam kalibrasyon görüntüsü üzerinde her iki modelin ham skorları hesaplanır:
   $$C_{\text{wrn}} = \{s_{\text{wrn}}(X_1^{\text{cal}}), \dots, s_{\text{wrn}}(X_n^{\text{cal}})\}, \quad C_{\text{dino}} = \{s_{\text{dino}}(X_1^{\text{cal}}), \dots, s_{\text{dino}}(X_n^{\text{cal}})\}$$
   Buradan ortalama ve standart sapmalar çıkarılır:
   $$\mu_{\text{wrn}} = \frac{1}{n} \sum_{i=1}^n C_{\text{wrn}}[i], \quad \sigma_{\text{wrn}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (C_{\text{wrn}}[i] - \mu_{\text{wrn}})^2}$$
   $$\mu_{\text{dino}} = \frac{1}{n} \sum_{i=1}^n C_{\text{dino}}[i], \quad \sigma_{\text{dino}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (C_{\text{dino}}[i] - \mu_{\text{dino}})^2}$$

2. **Adım 2 — Kalibrasyon Birleşik Uyumsuzluk Skorları ($S^{\text{cal}}$):**
   Her bir kalibrasyon görüntüsü ($i=1, \dots, n$) için birleşik skor hesaplanır:
   $$S^{\text{cal}}[i] = \frac{1}{2} \left( \frac{C_{\text{wrn}}[i] - \mu_{\text{wrn}}}{\sigma_{\text{wrn}}} + \frac{C_{\text{dino}}[i] - \mu_{\text{dino}}}{\sigma_{\text{dino}}} \right)$$

3. **Adım 3 — Gelen Yeni Test Görüntüsünün ($x$) Skorlanması:**
   Yeni bir görüntü geldiğinde iki modelden $s_{\text{wrn}}(x)$ ve $s_{\text{dino}}(x)$ alınır:
   $$S(x) = \frac{1}{2} \left( \frac{s_{\text{wrn}}(x) - \mu_{\text{wrn}}}{\sigma_{\text{wrn}}} + \frac{s_{\text{dino}}(x) - \mu_{\text{dino}}}{\sigma_{\text{dino}}} \right)$$

4. **Adım 4 — Konformal p-Değeri ve Kusur Kararı:**
   $$p(x) = \frac{1 + \sum_{i=1}^{n} \mathbf{1}\{ S^{\text{cal}}[i] \ge S(x) \}}{n + 1} = \frac{1 + \sum_{i=1}^{40} \mathbf{1}\{ S^{\text{cal}}[i] \ge S(x) \}}{41}$$
   $$\text{Karar}(x) = \begin{cases} \text{KUSURLU (DEFECT)}, & p(x) \le 0.05 \\ \text{SAĞLAM (GOOD)}, & p(x) > 0.05 \end{cases}$$

---

## BÖLÜM 5: Kritik Kısıtlar ve Dikkat Edilmesi Gerekenler (Caveats)

1. **Küçük Kalibrasyon Boyutu ($n=40$) ve p-Değeri Çözünürlüğü:**
   - Kalibrasyon kümesinde $n=40$ görüntü bulunduğundan, konformal p-değeri sürekli değil, adımlı bir kuantum dağılımına sahiptir:
     $$p \in \left\{ \frac{1}{41}, \frac{2}{41}, \frac{3}{41}, \dots, \frac{41}{41} \right\} \approx \{ 0.0244, 0.0488, 0.0732, \dots, 1.0000 \}$$
   - Bu durum iki kritik sonuç doğurur:
     - **$\alpha = 0.01$ gibi daha sıkı bir yanlış alarm hedefi seçilemez;** çünkü mümkün olan en küçük p-değeri dahi $1/41 \approx 0.0244 > 0.01$'dir.
     - **$\alpha = 0.05$ eşiğinde yalnızca ilk iki kuantum ($1/41$ ve $2/41$) ret bölgesine girer.** Post-hoc p-birleştirme yöntemlerinin (Bonferroni ve Vovk-Wang) çökmesinin temel nedeni budur; çünkü p-değerlerini toplamak veya çarpmak, sonucu doğrudan $0.05$'in üzerine fırlatmaktadır. "Combine-then-calibrate" ise skorları önce birleştirdiği için bu kuantum kaybına uğramaz.

2. **metal_nut Kategorisinde Yetersiz Sağlam Test Görüntüsü ($N_{\text{neg}} = 22$, Test Setinde 11):**
   - MVTec AD `metal_nut` test kümesinde toplam 22 adet sağlam görüntü vardır; test splitinde ise yalnızca 11 adet sağlam görüntü yer almaktadır.
   - 11 sağlam görüntüde **tek bir yanlış alarm dahi verilmesi durumunda**, ampirik yanlış alarm oranı:
     $$\text{FPR} = \frac{1}{11} \approx 9.09\% > 5.00\%$$
     olmaktadır. Yani %5 yanlış alarm hedefinin tutturulabilmesi için sistemin bu sınıfta **tam olarak 0 adet yanlış pozitif** vermesi zorunludur. `combine-then-calibrate` bu sınıfta sıfır yanlış pozitif ile %100 kusur yakalamayı başarmıştır.
