import pickle  # Python nesnelerini seri hale getirmek ve saklamak için
from sklearn.svm import SVC  # Destek vektör sınıflandırıcı modeli
from sklearn.metrics import classification_report  # Sınıflandırma raporu için
import pandas as pd  # Veri çerçeveleri için
import os  # İşletim sistemi işlevleri için

path = os.path.dirname(__file__)  # Bu dosyanın dizinini alır
os.chdir(path)  # Çalışma dizinini bu dosyanın dizini olarak ayarlar

# MNIST veri kümesini yükler
def load_mnist():
    with open('mnist.pkl', 'rb') as f:
        mnist = pickle.load(f)
    return mnist['training_images'], mnist['training_labels'], mnist['test_images'], mnist['test_labels']

# Eğitim ve test verilerini yükler
train_x, train_y, test_x, test_y = load_mnist()

# Verileri pandas DataFrame'e dönüştürür
train_x, train_y, test_x, test_y = [pd.DataFrame(x) for x in [train_x, train_y, test_x, test_y]]

# Verileri 0-1 aralığına normalize eder
train_x = train_x / 255.0
test_x = test_x / 255.0

# SVC modelini oluşturur ve eğitir
svc = SVC()
svc.fit(train_x, train_y.values.flatten())

# Eğitilmiş modeli bir pickle dosyasına kaydeder
filename = "svm_model.pkl"
pickle.dump(svc, open(filename, 'wb'))

# Test verileri üzerinde tahminler yapar
y_pred = svc.predict(test_x)

# Sınıflandırma raporunu yazdırır
print(classification_report(test_y, y_pred))
