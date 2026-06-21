# MNIST Digit Recognition with Random Forest & Gradio

Bu proje, makine öğrenmesi algoritmalarının web tabanlı bir kullanıcı arayüzü ile nasıl entegre edilebileceğini gösteren uçtan uca (end-to-end) bir uygulamadır. 

## 🧠 Proje Mimarisi ve Matematiksel Yaklaşım
Geleneksel MNIST el yazısı rakam veri seti kullanılarak, 28x28 piksellik (784 boyutlu vektör) görüntüler üzerinde bir sınıflandırma modeli eğitilmiştir. Çizimden alınan ham veriler matrislere dönüştürülüp `[0, 1]` aralığında normalize edildikten sonra **Random Forest Classifier** algoritması ile analiz edilmiştir.

## 🛠️ Kullanılan Teknolojiler
* **Model Eğitimi & Veri İşleme:** Python, Scikit-learn, NumPy, PIL
* **Kullanıcı Arayüzü (UI):** Gradio
* **Versiyon Kontrol:** Git & GitHub

## 🚀 Nasıl Çalıştırılır?
Projeyi lokalinizde çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Repoyu klonlayın: 
`git clone https://github.com/[KULLANICI_ADIN]/[REPO_ADIN].git`

2. Gerekli kütüphaneleri yükleyin: 
`pip install -r requirements.txt`

3. Uygulamayı başlatın: 
`python app.py`
