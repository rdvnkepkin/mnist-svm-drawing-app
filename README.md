# ✍️ İki Basamaklı El Yazısı Rakam Tanıma (Gradio & SVM)

Bu proje, makine öğrenmesi algoritmalarının kullanıcı dostu bir web arayüzü ile nasıl entegre edilebileceğini gösteren uçtan uca (end-to-end) bir analitik çözüm uygulamasıdır. Kullanıcıların çizdiği rakamları eşzamanlı olarak işleyip analiz eden çift panelli bir mimariye sahiptir.

## 🧠 Mimari ve Matematiksel Yaklaşım
Proje, geleneksel MNIST veri seti üzerinde eğitilmiş bir **Support Vector Machine (SVM)** modeli kullanılarak geliştirilmiştir. Modelin her seferinde baştan eğitilmesini önlemek amacıyla, eğitilmiş model bir `.pkl` dosyası olarak sisteme entegre edilmiş ve uygulamanın açılış süresi saniyelere indirilmiştir.

Kullanıcıdan alınan çizimler doğrudan işlenmez; arka planda doğrusal cebir kurallarına uygun bir veri dönüşüm hattından (pipeline) geçer:
* Ham çizim verisi 28x28 boyutlarında matrislere indirgenir.
* Siyah-beyaz renk uzayına (L) çevrilerek renkleri tersine çevrilir.
* Veriler 1x784 boyutlu vektörlere dönüştürülüp `[0, 1]` aralığında normalize edilir.

## 🛡️ Hata Yönetimi (Exception Handling) ve Kayıp Veri Kontrolü
Sistem, gerçek dünya senaryolarında sıklıkla karşılaşılan **"Kayıp Veri" (Missing Data)** durumlarına karşı defansif bir yapıda kurgulanmıştır:
* **Boş Matris Filtresi:** Tuval üzerine çizim yapılıp silindiğinde ortaya çıkan sıfır vektörleri (0 matris toplamı) matematiksel olarak tespit edilir ve modelin çökmeksizin bu durumu pas geçmesi sağlanır.
* **Dinamik Geri Bildirim:** Panellerden yalnızca birine veri girildiğinde sistem Null Pointer hataları fırlatmak yerine, yalnızca çizim yapılan paneli analiz ederek kullanıcıya spesifik geri bildirim ("Tek Rakam Algılandı") sunar.

## 🛠️ Kullanılan Teknolojiler
* **Makine Öğrenmesi & Veri İşleme:** Python, Scikit-learn (SVM), NumPy, PIL
* **Kullanıcı Arayüzü (UI):** Gradio (Blocks Mimarisi)

## 🚀 Çalıştırma Talimatları
1. Repoyu klonlayın: `git clone https://github.com/rdvnkepkin/mnist-svm-drawing-app.git`
2. Gereksinimleri yükleyin: `pip install -r requirements.txt`
3. Uygulamayı başlatın: `python app.py`
