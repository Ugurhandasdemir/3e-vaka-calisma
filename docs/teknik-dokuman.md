# 3E Elektro Optik — Yapay Zekâ Destekli Kalite Kontrol PoC

**Uygulama:** https://qc.ugurhandasdemir.com (yedek adres: https://huggingface.co/spaces/ugurhandasdemir/3e-vaka-calisma)
**Kaynak kod:** https://github.com/Ugurhandasdemir/3e-vaka-calisma

### 1. Problem ve çözüm
Kalite kontrol departmanında günde ortalama 85 ürün elle muayene ediliyor. Kayıtlar kâğıt form, Excel ve ERP arasında tekrar tekrar giriliyor, iki yıllık 1.200 kusur görseli etiketsiz duruyor ve son altı ayda üç kusur müşteriye ulaşmış. Bu PoC'de muayene süreci iki aşamaya ayrıldı. Üretim İçi Muayene aşamasında yüzey çiziği, kaplama ve konektör/montaj gibi görünür kusurlar yapay zekâ ile incelenir. Üretim Sonrası Test aşamasında ise optik eksen hizası, görüntü keskinliği ve vida gevşemesi gibi ölçüm gerektiren kontroller yapılır. İki aşama aynı seri numarası ve kayıt üzerinden ilerler ve bir ürün ancak iki aşamadan da geçerse "sevke hazır" olur. Yapay zekâ karar vermez, karar önerir. Son karar her zaman muayene uzmanındadır.

### 2. Kullanılan yapay zekâ modelleri ve teknolojiler
Sistem hibrit olarak tasarlandı ve anomali tespiti, kusur tespiti ve görsel dil modeli (VLM) aşamalarından oluşuyor. Sistemin merkezinde etiket gerektirmeyen bir anomali tespiti var. Model yalnızca sağlam parçaların görüntülerinden "normal"in nasıl göründüğünü öğreniyor ve yeni görseldeki sapmayı bir ısı haritasıyla gösteriyor. Bunun için PatchCore ve DINOv2 tabanlı AnomalyDINO birlikte çalışıyor. İki model farklı hatalara karşı hassas olduğu için birinin kaçırdığını diğeri yakalayabiliyor. İki modelin skorları birleştiriliyor ve sağlam kalibrasyon görselleriyle istatistiksel olarak kalibre ediliyor. Böylece "bu parça normalden ne kadar farklı" sorusu, yanlış alarm oranı kontrol altında tutulan bir güven değerine dönüşüyor.

Kusurun türünü (çizik, kaplama, konektör/montaj) eğitilmiş bir YOLO11n nesne tespiti modeli buluyor. Bu modeli Google Colab'da, MVTec veri setindeki gerçek kusur görselleriyle eğittim. Geliştirme yaptığım bilgisayarda 4 GB VRAM bulunduğu için görsel dil modelini yerelde çalıştırmak yerine Gemini (3.5 Flash-Lite) API'si üzerinden kullandım. Gemini ısı haritasına ve YOLO'nun bulduğu kutulara bakarak uzmanın okuyacağı bir gerekçe yazıyor. Aynı soru üç kez soruluyor ve cevapların ne kadar tutarlı olduğu da hesaba katılıyor.

Üç motorun sonucu kural tabanlı bir karar katmanında birleşiyor. Böylece bir aşamanın kaçırdığı detayı diğer aşamalar yakalayabiliyor. Motorlar aynı sonuca varıyorsa sistem otomatik olarak kabul ya da ret öneriyor. Motorlar çelişiyorsa ya da sonuç sınırdaysa parça insan incelemesine gönderiliyor. Termal kamera modülleri daha önce geri çağırmaya yol açtığı için bu grupta otomatik kabul hiç verilmiyor.

Hatalı durumlarda operatör kusuru görsel üzerinde bir kutu çizerek işaretleyebiliyor. Kutunun içinde SAM (Segment Anything) modeli çalışıyor ve kusurun maskesini çıkarıyor. Böylece tek işlemle hem kutu hem de maske etiketi elde ediliyor ve bu etiketler veri setine eklenerek modelin yeniden eğitiminde kullanılabiliyor.

Üretim sonrası testlerde (optik eksen, keskinlik, tork işareti) yapay zekâ yerine ölçüme dayalı görüntü işleme kullandım. Bu kontrollerde sonucun bir sayı olarak, tekrarlanabilir ve açıklanabilir şekilde verilmesi gerekiyor. Optik eksen testi, kolimatör hedefinin görüntüdeki merkezini piksel altı hassasiyetle bulup sapmayı açıya çeviriyor ve titreşim testinden önce ve sonra ölçülen değerleri karşılaştırabiliyor. Keskinlik testi standart eğik kenar yöntemiyle (ISO 12233) MTF değerini hesaplıyor. Tork işareti testi ise vida başına çekilen boya çizgisinin kayıp kaymadığına bakarak gevşemeyi tespit ediyor.

### 3. Veri yaklaşımı
Şirket verisine erişimim olmadığı için MVTec AD veri setini temsili veri olarak kullandım. Metal somun görselleri optik lens grubunu, transistor görselleri termal kamera modülünü, kablo görselleri de gözetleme ünitesini temsil ediyor. Anomali modeli her ürün grubu için yalnızca 60 sağlam referans görselle kuruluyor ve 40 sağlam görselle kalibre ediliyor, yani başlangıçta etiketli veriye ihtiyaç duymuyor. Bu, şirketteki etiketsiz 1.200 görsel sorununa doğrudan bir cevap. Sistem kullanıldıkça operatörün kararları ve çizdiği etiketler birikiyor ve bunlar dışa aktarılarak YOLO modelinin yeniden eğitilmesinde kullanılabiliyor. Ölçüm modüllerini, sonucu önceden bilinen sentetik test görüntüleriyle doğruladım.

### 4. Temel teknik mimari
```
[Görsel + seri no] → Anomali tespiti (PatchCore + DINOv2) → ısı haritası ve güven değeri
                   → YOLO11n (kusur türü ve konumu) → Gemini (gerekçe)
                   → Karar katmanı: KABUL / İNSAN İNCELEMESİ / RET → Uzmanın kararı ve etiketi (SAM)
[Son test]         → Optik eksen (boresight) · Keskinlik (MTF) · Tork işareti
Kalite kapısı      → görsel muayene KABUL + son test GEÇTİ ise SEVKE HAZIR
Kayıt              → görsel, model sürümü, AI önerisi, uzman kararı, kullanıcı, zaman → CSV ve etiket dışa aktarma
```
Uygulama Python ve Gradio ile yazıldı, Docker ile paketlendi. Ana adres kendi sunucumda Cloudflare Tunnel üzerinden yayında ve kayıtlar kalıcı olarak saklanıyor. Hugging Face üzerinde de yedek bir kopyası çalışıyor. Mimarinin ayrıntılı şeması ekteki "3E-Mimari-Semasi.png" dosyasındadır.

### 5. Ölçülen sonuçlar
Sistemi MVTec'in 365 gerçek test görseli üzerinde ölçtüm. Görselleri ayar ve test olarak ikiye ayırdım, eşikleri yalnızca ayar kısmında belirleyip sonuçları test kısmında raporladım.

| Ölçüm | Sonuç |
|---|---|
| Anomali tespiti başarısı (AUROC) | Tek model 0,988, iki model birlikte **0,995** |
| Kusur yakalama ve yanlış alarm | Tek model %91,2 ve %4,3, iki model birlikte **%92,9 ve %0** |
| Uçtan uca karar (test kısmı) | **Kaçan kusur 0, yanlış ret 0**, insan incelemesine giden %20,2 |
| YOLO11n (mAP50) | 0,814 (uygulamadaki ürün gruplarında 0,779) |
| Ölçüm modülleri (sentetik hedefler) | Optik eksende en fazla 0,06 piksel, MTF'te en fazla %1,9 hata, tork işaretinde 19/19 doğru karar |
| Analiz süresi | Görsel başına yaklaşık 4–8 saniye |

### 6. Güvenlik yaklaşımı
API anahtarı kodda ya da repoda değil, sunucunun gizli ayarlarında tutuluyor. Yüklenen görsellerin boyutu sınırlandırılıyor ve aynı anda işlenen istek sayısı kısıtlanıyor. Her karar izlenebilir şekilde kaydediliyor. Hangi görselin, hangi model sürümüyle, ne önerildiği ve uzmanın ne karar verdiği, kim tarafından ve ne zaman verildiği tutuluyor. Bu yapı savunma sanayindeki izlenebilirlik beklentisiyle uyumlu. Sistem emin olmadığında "AI önerisini onayla" seçeneği kapanıyor ve uzmanın kendisi karar vermek zorunda kalıyor. Teknik yetersizliklerden dolayı vaka görseller Gemini API'sine gönderiliyor. Gerçek üretimde savunma verisinin kurum dışına çıkmaması gerektiği için bu kısmın yerel bir modelle değiştirilmesi gerekiyor.

### 7. Uygulamanın sınırlılıkları
Kullanılan veri temsili. MVTec görselleri gerçek lens ya da termal modül değil ve şeffaf, yansıtıcı yüzeylerde başarının düşmesi beklenir. YOLO modeli MVTec test görsellerinin bir kısmıyla eğitildiği için uçtan uca sonuçlar dedektör açısından olduğundan iyi görünüyor olabilir. Kalibrasyon için ürün grubu başına yalnızca 40 sağlam görsel kullanıldı. Bu sayı arttıkça insan incelemesine giden parça oranı düşecektir, güvenilir bir %1 yanlış alarm oranı için grup başına yaklaşık 300 görsel gerekiyor. Ölçüm modülleri yalnızca sentetik görüntülerle doğrulandı ve keskinlik ile tork sonuçları henüz kalite kapısına bağlanmadı. Konektör tutma kuvveti ve termal gürültü (NETD) gibi ölçümler ise özel test donanımı gerektiriyor. Son olarak VLM'in verdiği güven değeri kalibre değil, bu yüzden karar katmanında yardımcı rol üstleniyor.

### 8. Gerçek üretim ortamına geçişte yapılması gerekenler
1. Kontrollü aydınlatma ve sabit fikstürle gerçek ürün görüntüleri toplanmalı, 1.200 görsel SAM ile ön etiketlenip uzman onayından geçirilmeli ve her ürün grubu için en az 300 sağlam kalibrasyon görseli hazırlanmalı.
2. Veri kurum dışına çıkmaması için Gemini yerine kurum içinde çalışan küçük bir görsel dil modeli (örneğin Qwen-VL tabanlı) kullanılmalı, sistem yetkilendirme ve değiştirilemez kayıt altyapısıyla kurumun kendi sunucusunda çalışmalı.
3. YOLO gerçek verilerle segmentasyon modeli olarak yeniden eğitilmeli, karar katmanının ağırlıkları zamanla uzman kararlarından öğrenilmeli.
4. Son test istasyonuna kolimatör, ısıtılmış hedef ve kenar hedefi entegre edilmeli, tork kontrolü akıllı sıkma aletlerinin tork–açı verisiyle desteklenmeli.
5. Sistem bir süre mevcut süreçle paralel çalıştırılarak sonuçları karşılaştırılmalı, ölçüm sistemi analizi (MSA) yapılmalı ve ERP/MES ile seri numarası ve iş emri entegrasyonu kurulmalı.

---
*Geliştirme sürecinde yapay zekâ kodlama asistanlarından (Claude, Gemini) yararlandım. Mimari, yöntem seçimleri ve deney tasarımı bana aittir.*
