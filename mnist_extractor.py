from torchvision import datasets
# Bu komut dosyaları otomatik olarak indirir ve klasörler
train_set = datasets.MNIST('./data', train=True, download=True)
import pickle
import numpy as np
import os

def save_mnist():
    # Bu dosyanın dizinini alır
    path = os.path.dirname(os.path.abspath(__file__))
    # Çalışma dizinini bu dosyanın dizini olarak ayarlar
    os.chdir(path)

    # MNIST veri setinin dosya adlarını içeren bir liste
    filename = [
        ["training_images", "train-images.idx3-ubyte"],
        ["test_images", "t10k-images.idx3-ubyte"],
        ["training_labels", "train-labels.idx1-ubyte"],
        ["test_labels", "t10k-labels.idx1-ubyte"]
    ]

    # ÖNCE KONTROL: Gerekli 4 dosya da klasörde var mı?
    for name in filename:
        file_path = os.path.join(path, name[1])
        if not os.path.exists(file_path):
            print(f"HATA: {name[1]} dosyası klasörde eksik!")
            print(f"Lütfen '{path}' klasörüne bu dosyayı ekleyin.")
            return # Eksik dosya varsa fonksiyonu burada bitir, aşağıya geçme!

    mnist = {}  # Boş bir sözlük oluşturur

    print("Dosyalar doğrulandı, veri yükleniyor...")

    # Görüntü verilerini okur (İlk 2 dosya)
    for name in filename[:2]:
        file_path = os.path.join(path, name[1])
        with open(file_path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

    # Etiket verilerini okur (Son 2 dosya)
    for name in filename[-2:]:
        file_path = os.path.join(path, name[1])
        with open(file_path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    # Elde edilen mnist sözlüğünü pickle dosyasına kaydeder
    print("Veriler 'mnist.pkl' dosyasına kaydediliyor...")
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
        
    print("İşlem başarıyla tamamlandı. 'mnist.pkl' oluşturuldu.")

if __name__ == "__main__":
    save_mnist()