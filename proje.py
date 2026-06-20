import os
import sys
import PyQt5

# BAŞLANGIÇ: Daha Güvenli Qt Hata Çözümü
plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
# BİTİŞ

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PIL import Image, ImageOps
import pickle

path = os.path.dirname(__file__)
os.chdir(path)

class MainWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.loaded_model = pickle.load(open('svm_model.pkl', 'rb'))

        self.initUI()
    
    def initUI(self):
        self.container = QtWidgets.QVBoxLayout()
        self.container.setContentsMargins(0, 0, 0, 0)

        self.label1 = QtWidgets.QLabel()  # İlk çizim alanı
        self.label2 = QtWidgets.QLabel()  # İkinci çizim alanı
        canvas1 = QtGui.QPixmap(300, 300)  # 300x300 piksel boyutunda bir alan oluşturulur
        canvas1.fill(QtGui.QColor("black"))  # Başlangıçta siyah bir arka plan oluşturulur
        canvas2 = QtGui.QPixmap(300, 300)  # 300x300 piksel boyutunda bir alan oluşturulur
        canvas2.fill(QtGui.QColor("black"))  # Başlangıçta siyah bir arka plan oluşturulur
        self.label1.setPixmap(canvas1)
        self.label2.setPixmap(canvas2)
        self.last_x1, self.last_y1 = None, None  # İlk çizim alanı için fare konumunu takip etmek için değişkenler
        self.last_x2, self.last_y2 = None, None  # İkinci çizim alanı için fare konumunu takip etmek için değişkenler

        self.prediction1 = QtWidgets.QLabel('Tahmin: ...')  # İlk çizim alanı için tahmin sonucunu gösterecek etiket
        self.prediction1.setFont(QtGui.QFont('Monospace', 20))  # Yazı tipi ve boyutunu ayarlar
        self.prediction2 = QtWidgets.QLabel('Tahmin: ...')  # İkinci çizim alanı için tahmin sonucunu gösterecek etiket
        self.prediction2.setFont(QtGui.QFont('Monospace', 20))  # Yazı tipi ve boyutunu ayarlar

        self.button_sil1 = QtWidgets.QPushButton('TEMİZLE')  # İlk çizim alanını temizleyen buton
        self.button_sil1.setStyleSheet("""
            QPushButton {
                background-color: #cc0000;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff0000;
                color: black;
            }
            QPushButton:pressed {
                background-color: maroon;
            }
        """)
        self.button_sil1.clicked.connect(lambda: self.sil_canvas1(canvas_number=1))

        self.button_sil2 = QtWidgets.QPushButton('TEMİZLE')  # İkinci çizim alanını temizleyen buton
        self.button_sil2.setStyleSheet("""
            QPushButton {
                background-color: #cc0000;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff0000;
                color: black;
            }
            QPushButton:pressed {
                background-color: maroon;
            }
        """)
        self.button_sil2.clicked.connect(lambda: self.sil_canvas2(canvas_number=2))

        self.button_tahminmet1 = QtWidgets.QPushButton('TAHMİN ET')  # İlk çizim alanındaki sayıyı tahmin eden buton
        self.button_tahminmet1.clicked.connect(self.predict1)
        self.button_tahminmet1.setStyleSheet("""
            QPushButton {
                background-color: #009900;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1aff1a;
                color: black;
            }
            QPushButton:pressed {
                background-color: maroon;
            }
        """)
        
        self.button_tahminmet2 = QtWidgets.QPushButton('TAHMİN ET')  # İkinci çizim alanındaki sayıyı tahmin eden buton
        self.button_tahminmet2.clicked.connect(self.predict2)
        self.button_tahminmet2.setStyleSheet("""
            QPushButton {
                background-color: #009900;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1aff1a;
                color: black;
            }
            QPushButton:pressed {
                background-color: maroon;
            }
        """)

        self.final_prediction = QtWidgets.QLabel('Tahminler: ...')  # Tahminlerin birleşik sonucunu gösteren label
        self.final_prediction.setFont(QtGui.QFont('Monospace', 16))  # Yazı tipi ve boyutunu ayarlar
        self.final_prediction.setAlignment(QtCore.Qt.AlignHCenter)  # Metni yatayda ortalar

        self.container1 = QtWidgets.QVBoxLayout()  # İlk çizim alanı için dikey düzen
        self.container1.addWidget(self.label1)
        self.container1.addWidget(self.prediction1, alignment = QtCore.Qt.AlignHCenter)
        self.container1.addWidget(self.button_sil1)
        self.container1.addWidget(self.button_tahminmet1)
        
        self.container2 = QtWidgets.QVBoxLayout()  # İkinci çizim alanı için dikey düzen
        self.container2.addWidget(self.label2)
        self.container2.addWidget(self.prediction2, alignment = QtCore.Qt.AlignHCenter)
        self.container2.addWidget(self.button_sil2)
        self.container2.addWidget(self.button_tahminmet2)

        self.hbox = QtWidgets.QHBoxLayout()  # İki çizim alanını yatay düzende birleştiren düzen
        self.hbox.addLayout(self.container1)
        self.hbox.addLayout(self.container2)

        self.container.addWidget(self.final_prediction)  # Tahminlerin birleşik sonucunu gösteren label'i ana düzene ekler
        self.container.addLayout(self.hbox)  # İki çizim alanını ve kontrolleri ana düzene ekler

        self.setLayout(self.container)
        self.label2.mouseMoveEvent = self.mouseMoveEvent1
        self.label2.mouseReleaseEvent = self.mouseReleaseEvent1

    # İlk çizim alanını temizleyen metot
    def sil_canvas1(self, canvas_number=1):
        if canvas_number == 1:
            self.label1.pixmap().fill(QtGui.QColor('#000000'))  # İlk çizim alanını temizler
            self.prediction1.setText('Tahmin:...')  # Tahmin etiketini sıfırlar
        elif canvas_number == 2:
            self.label2.pixmap().fill(QtGui.QColor('#000000'))  # İkinci çizim alanını temizler
            self.prediction2.setText('Tahmin:...')  # Tahmin etiketini sıfırlar
        self.final_prediction.setText('Tahminler: ...')  # Birleşik tahminleri sıfırlar
        self.update()

    # İkinci çizim alanını temizleyen metot
    def sil_canvas2(self, canvas_number=2):
        if canvas_number == 2:
            self.label2.pixmap().fill(QtGui.QColor('#000000'))  # İkinci çizim alanını temizler
            self.prediction2.setText('Tahmin:...')  # Tahmin etiketini sıfırlar
        self.final_prediction.setText('Tahminler: ...')  # Birleşik tahminleri sıfırlar
        self.update()

    # İlk çizim alanındaki rakamı tahmin eden metot
    def predict1(self):
        s = self.label1.pixmap().toImage().bits().asarray(300 * 300 * 4)
        arr = np.frombuffer(s, dtype=np.uint8).reshape((300, 300, 4))
        arr = np.array(ImageOps.grayscale(Image.fromarray(arr).resize((28,28), Image.BICUBIC)))
        arr = (arr/255.0).reshape(1, -1)
        self.prediction1.setText('Tahmin: '+str(self.loaded_model.predict(arr)[0]))
        self.update_final_prediction()  # Tahminlerin birleşik sonucunu güncelle

    # İkinci çizim alanındaki rakamı tahmin eden metot
    def predict2(self):
        s = self.label2.pixmap().toImage().bits().asarray(300 * 300 * 4)
        arr = np.frombuffer(s, dtype=np.uint8).reshape((300, 300, 4))
        arr = np.array(ImageOps.grayscale(Image.fromarray(arr).resize((28,28), Image.BICUBIC)))
        arr = (arr/255.0).reshape(1, -1)
        self.prediction2.setText('Tahmin: '+str(self.loaded_model.predict(arr)[0]))
        self.update_final_prediction()  # Tahminlerin birleşik sonucunu güncelle
    
    def update_final_prediction(self):
        # Tahminlerin birleşik sonucunu günceller
        tahmin1 = self.prediction1.text().split(':')[-1].strip()  # İlk tahmini al
        tahmin2 = self.prediction2.text().split(':')[-1].strip()  # İkinci tahmini al
        self.final_prediction.setText(f'Tahminler: {tahmin1}{tahmin2}')  # Birleşik sonucu göster

    # İlk çizim alanındaki fare hareketini takip eden olay yöntemi
    def mouseMoveEvent(self, e):
        # İlk çizim alanındaki fare hareketini takip eder
        if self.last_x1 is None:
            self.last_x1 = e.x()
            self.last_y1 = e.y()
            return

        painter = QtGui.QPainter(self.label1.pixmap())

        p = painter.pen()
        p.setWidth(15)
        self.pen_color = QtGui.QColor('#ffffff')
        p.setColor(self.pen_color)
        painter.setPen(p)

        painter.drawLine(self.last_x1, self.last_y1, e.x(), e.y())
        painter.end()
        self.update()

        self.last_x1 = e.x()
        self.last_y1 = e.y()
    
    def mouseReleaseEvent(self, e):
        self.last_x1 = None
        self.last_y1 = None

    # İkinci çizim alanındaki fare hareketini takip eden olay yöntemi    
    def mouseMoveEvent1(self, e1):
            # ikinici çizim alanındaki fare hareketini takip eder
        if self.last_x2 is None:
            self.last_x2 = e1.x()
            self.last_y2 = e1.y()
            return

        painter1 = QtGui.QPainter(self.label2.pixmap())

        p1 = painter1.pen()
        p1.setWidth(15)
        self.pen_color1 = QtGui.QColor('#ffffff')
        p1.setColor(self.pen_color1)
        painter1.setPen(p1)

        painter1.drawLine(self.last_x2, self.last_y2, e1.x(), e1.y())
        painter1.end()
        self.update()

        self.last_x2 = e1.x()
        self.last_y2 = e1.y()
    
    def mouseReleaseEvent1(self, e1):
        self.last_x2 = None
        self.last_y2 = None

# Ana pencere sınıfı
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Ana widget oluşturuluyor ve pencereye atanıyor
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)

# Uygulamanın başlatılması
if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    mainApp = MainWindow()
    mainApp.setWindowTitle('SAYI ÇİZ')
    mainApp.show()
    sys.exit(app.exec_())
