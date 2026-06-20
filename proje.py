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
        
        # ÇÖZÜM 1: Çerçevelerin esnemesini engelliyor, boyutu 300x300'e çiviliyoruz.
        self.label1.setFixedSize(300, 300)
        self.label2.setFixedSize(300, 300)

        self.last_x1, self.last_y1 = None, None  
        self.last_x2, self.last_y2 = None, None  

        self.prediction1 = QtWidgets.QLabel('Tahmin: ...')  
        self.prediction1.setFont(QtGui.QFont('Monospace', 20))  
        self.prediction2 = QtWidgets.QLabel('Tahmin: ...')  
        self.prediction2.setFont(QtGui.QFont('Monospace', 20))  

        self.button_sil1 = QtWidgets.QPushButton('TEMİZLE')  
        self.button_sil1.setStyleSheet("""
            QPushButton { background-color: #cc0000; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background-color: #ff0000; color: black; }
            QPushButton:pressed { background-color: maroon; }
        """)
        self.button_sil1.clicked.connect(lambda: self.sil_canvas(1))

        self.button_sil2 = QtWidgets.QPushButton('TEMİZLE')  
        self.button_sil2.setStyleSheet("""
            QPushButton { background-color: #cc0000; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background-color: #ff0000; color: black; }
            QPushButton:pressed { background-color: maroon; }
        """)
        self.button_sil2.clicked.connect(lambda: self.sil_canvas(2))

        self.button_tahminmet1 = QtWidgets.QPushButton('TAHMİN ET')  
        self.button_tahminmet1.clicked.connect(self.predict1)
        self.button_tahminmet1.setStyleSheet("""
            QPushButton { background-color: #009900; color: white; border-radius: 10px; padding: 10px 20px; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background-color: #1aff1a; color: black; }
            QPushButton:pressed { background-color: maroon; }
        """)
        
        self.button_tahminmet2 = QtWidgets.QPushButton('TAHMİN ET')  
        self.button_tahminmet2.clicked.connect(self.predict2)
        self.button_tahminmet2.setStyleSheet("""
            QPushButton { background-color: #009900; color: white; border-radius: 10px; padding: 10px 20px; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background-color: #1aff1a; color: black; }
            QPushButton:pressed { background-color: maroon; }
        """)

        self.final_prediction = QtWidgets.QLabel('Tahminler: ...')  
        self.final_prediction.setFont(QtGui.QFont('Monospace', 16))  
        self.final_prediction.setAlignment(QtCore.Qt.AlignHCenter)  

        self.container1 = QtWidgets.QVBoxLayout()  
        self.container1.addWidget(self.label1, alignment=QtCore.Qt.AlignHCenter)
        self.container1.addWidget(self.prediction1, alignment=QtCore.Qt.AlignHCenter)
        self.container1.addWidget(self.button_sil1)
        self.container1.addWidget(self.button_tahminmet1)
        
        self.container2 = QtWidgets.QVBoxLayout()  
        self.container2.addWidget(self.label2, alignment=QtCore.Qt.AlignHCenter)
        self.container2.addWidget(self.prediction2, alignment=QtCore.Qt.AlignHCenter)
        self.container2.addWidget(self.button_sil2)
        self.container2.addWidget(self.button_tahminmet2)

        self.hbox = QtWidgets.QHBoxLayout()  
        self.hbox.addLayout(self.container1)
        self.hbox.addLayout(self.container2)

        self.container.addWidget(self.final_prediction)  
        self.container.addLayout(self.hbox)  

        self.setLayout(self.container)
        
        # ÇÖZÜM 2: Fare okuma olaylarını pencereye değil, DOĞRUDAN kendi çizim alanlarına bağlıyoruz
        self.label1.mouseMoveEvent = self.fareHareket1
        self.label1.mouseReleaseEvent = self.fareBirak1
        
        self.label2.mouseMoveEvent = self.fareHareket2
        self.label2.mouseReleaseEvent = self.fareBirak2

    def sil_canvas(self, canvas_number):
        if canvas_number == 1:
            self.label1.pixmap().fill(QtGui.QColor('#000000'))  
            self.prediction1.setText('Tahmin:...')  
        elif canvas_number == 2:
            self.label2.pixmap().fill(QtGui.QColor('#000000'))  
            self.prediction2.setText('Tahmin:...')  
        self.final_prediction.setText('Tahminler: ...')  
        self.update()

    def predict1(self):
        s = self.label1.pixmap().toImage().bits().asarray(300 * 300 * 4)
        arr = np.frombuffer(s, dtype=np.uint8).reshape((300, 300, 4))
        arr = np.array(ImageOps.grayscale(Image.fromarray(arr).resize((28,28), Image.BICUBIC)))
        arr = (arr/255.0).reshape(1, -1)
        self.prediction1.setText('Tahmin: '+str(self.loaded_model.predict(arr)[0]))
        self.update_final_prediction()  

    def predict2(self):
        s = self.label2.pixmap().toImage().bits().asarray(300 * 300 * 4)
        arr = np.frombuffer(s, dtype=np.uint8).reshape((300, 300, 4))
        arr = np.array(ImageOps.grayscale(Image.fromarray(arr).resize((28,28), Image.BICUBIC)))
        arr = (arr/255.0).reshape(1, -1)
        self.prediction2.setText('Tahmin: '+str(self.loaded_model.predict(arr)[0]))
        self.update_final_prediction()  
    
    def update_final_prediction(self):
        tahmin1 = self.prediction1.text().split(':')[-1].strip()  
        tahmin2 = self.prediction2.text().split(':')[-1].strip()  
        if tahmin1 != "..." or tahmin2 != "...":
            t1 = "" if tahmin1 == "..." else tahmin1
            t2 = "" if tahmin2 == "..." else tahmin2
            self.final_prediction.setText(f'Tahminler: {t1}{t2}')

    # --- 1. ÇİZİM ALANI FARE KONTROLLERİ ---
    def fareHareket1(self, e):
        if self.last_x1 is None:
            self.last_x1 = e.x()
            self.last_y1 = e.y()
            return

        painter = QtGui.QPainter(self.label1.pixmap())
        p = painter.pen()
        p.setWidth(15)
        p.setColor(QtGui.QColor('#ffffff'))
        painter.setPen(p)
        painter.drawLine(self.last_x1, self.last_y1, e.x(), e.y())
        painter.end()
        self.label1.update() # Ana pencereyi değil, sadece etiketi günceller

        self.last_x1 = e.x()
        self.last_y1 = e.y()
    
    def fareBirak1(self, e):
        self.last_x1 = None
        self.last_y1 = None

    # --- 2. ÇİZİM ALANI FARE KONTROLLERİ ---
    def fareHareket2(self, e):
        if self.last_x2 is None:
            self.last_x2 = e.x()
            self.last_y2 = e.y()
            return

        painter = QtGui.QPainter(self.label2.pixmap())
        p = painter.pen()
        p.setWidth(15)
        p.setColor(QtGui.QColor('#ffffff'))
        painter.setPen(p)
        painter.drawLine(self.last_x2, self.last_y2, e.x(), e.y())
        painter.end()
        self.label2.update() # Ana pencereyi değil, sadece etiketi günceller

        self.last_x2 = e.x()
        self.last_y2 = e.y()
    
    def fareBirak2(self, e):
        self.last_x2 = None
        self.last_y2 = None

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    mainApp = MainWindow()
    mainApp.setWindowTitle('SAYI ÇİZ')
    mainApp.show()
    sys.exit(app.exec_())
