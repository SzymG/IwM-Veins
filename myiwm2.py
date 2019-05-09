import sys

from PIL.ImageQt import ImageQt
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import imageio
import numpy as np
import cv2
from scipy.misc import toimage

class Window(QtWidgets.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.window = QtWidgets.QMainWindow()
        self.setGeometry(0, 0, 1000, 745)
        self.setWindowTitle("Vein verification")

        self.imgAs2DArray = 0
        self.clahe = 0

        self.label = QtWidgets.QLabel(self)

        self.label1 = QtWidgets.QLabel(self)
        self.label2 = QtWidgets.QLabel(self)
        self.label3 = QtWidgets.QLabel(self)

        self.label1_miara = QtWidgets.QLabel(self)
        self.label2_miara = QtWidgets.QLabel(self)
        self.label3_miara = QtWidgets.QLabel(self)
        self.label4_miara = QtWidgets.QLabel(self)
        self.label5_miara = QtWidgets.QLabel(self)

        self.label1_miara_wart = QtWidgets.QLabel(self)
        self.label2_miara_wart = QtWidgets.QLabel(self)
        self.label3_miara_wart = QtWidgets.QLabel(self)
        self.label4_miara_wart = QtWidgets.QLabel(self)
        self.label5_miara_wart = QtWidgets.QLabel(self)

        self.image_label1 = QtWidgets.QLabel(self)
        self.image_label2 = QtWidgets.QLabel(self)
        self.image_label3 = QtWidgets.QLabel(self)

        self.initGUI()


    def initGUI(self):

        self.btn_prep = QtWidgets.QPushButton("Preprocessing", self)
        self.btn_prep.setGeometry(30, 475, 300, 50)
        self.btn_prep.setStyleSheet("font-size: 18px;")
        self.btn_prep.clicked.connect(self.preprocessing)
        self.btn_prep.setEnabled(False)

        self.btn_start = QtWidgets.QPushButton("Start!", self)
        self.btn_start.setGeometry(350, 475, 300, 50)
        self.btn_start.setStyleSheet("font-size: 18px;")
        self.btn_start.clicked.connect(self.start)
        self.btn_start.setEnabled(False)

        self.btn_post = QtWidgets.QPushButton("Postprocessing", self)
        self.btn_post.setGeometry(670, 475, 300, 50)
        self.btn_post.setStyleSheet("font-size: 18px;")
        self.btn_post.clicked.connect(self.postprocessing)
        self.btn_post.setEnabled(False)

        self.btn_choose = QtWidgets.QPushButton("Wybierz obraz wejściowy", self)
        self.btn_choose.setGeometry(30, 25, 300, 50)
        self.btn_choose.setStyleSheet("font-size: 18px;")
        self.btn_choose.clicked.connect(self.choose_file)

        self.label.setGeometry(430, 540, 300, 50)
        self.label.setText("MIARY")
        self.label.setStyleSheet("font-size: 26px;")

        self.label1.setGeometry(50, 90, 280, 50)
        self.label1.setText("Obraz wejściowy/preprocessing")
        self.label1.setStyleSheet("font-size: 18px;")

        self.label2.setGeometry(400, 90, 280, 50)
        self.label2.setText("Przetwarzanie właściwe")
        self.label2.setStyleSheet("font-size: 18px;")

        self.label3.setGeometry(690, 90, 280, 50)
        self.label3.setText("Obraz wyjściowy/postprocessing")
        self.label3.setStyleSheet("font-size: 18px;")

        self.image_label1.setGeometry(30, 150, 300, 300)
        self.image_label1.setStyleSheet("border: 1px solid #000000;")

        self.image_label2.setGeometry(350, 150, 300, 300)
        self.image_label2.setStyleSheet("border: 1px solid #000000;")

        self.image_label3.setGeometry(670, 150, 300, 300)
        self.image_label3.setStyleSheet("border: 1px solid #000000;")

        self.label1_miara.setGeometry(30, 610, 100, 50)
        self.label1_miara.setText("PRECYZJA: ")
        self.label1_miara.setStyleSheet("font-size: 18px;")

        self.label1_miara_wart.setGeometry(130, 615, 80, 40)
        self.label1_miara_wart.setStyleSheet("font-size: 18px; border: 1px solid #000000;")

        self.label2_miara.setGeometry(220, 610, 100, 50)
        self.label2_miara.setText("CZUŁOŚĆ: ")
        self.label2_miara.setStyleSheet("font-size: 18px;")

        self.label2_miara_wart.setGeometry(310, 615, 80, 40)
        self.label2_miara_wart.setStyleSheet("font-size: 18px; border: 1px solid #000000;")

        self.label3_miara.setGeometry(410, 610, 100, 50)
        self.label3_miara.setText("TRAFNOŚĆ: ")
        self.label3_miara.setStyleSheet("font-size: 18px;")

        self.label3_miara_wart.setGeometry(510, 615, 80, 40)
        self.label3_miara_wart.setStyleSheet("font-size: 18px; border: 1px solid #000000;")

        self.label4_miara.setGeometry(610, 610, 120, 50)
        self.label4_miara.setText("SWOISTOŚĆ: ")
        self.label4_miara.setStyleSheet("font-size: 18px;")

        self.label4_miara_wart.setGeometry(725, 615, 80, 40)
        self.label4_miara_wart.setStyleSheet("font-size: 18px; border: 1px solid #000000;")

        self.label5_miara.setGeometry(30, 675, 450, 50)
        self.label5_miara.setText("ŚREDNIA ARYTMETYCZNA CZUŁOŚCI I SWOISTOŚCI: ")
        self.label5_miara.setStyleSheet("font-size: 18px;")

        self.label5_miara_wart.setGeometry(480, 680, 80, 40)
        self.label5_miara_wart.setStyleSheet("font-size: 18px; border: 1px solid #000000;")

        self.show()

    def choose_file(self):

        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')

        s1 = "/"
        s2 = name[0].split("/")[:-1]
        self.path = s1.join(s2)
        print(self.path)

        self.img_name = name[0].split("/")[-1]
        self.img_master_name = self.img_name.split(".")[0] + ".ah." + self.img_name.split(".")[1]
        self.img = imageio.imread(name[0])

        pixmap = QtGui.QPixmap(name[0])
        pixmap = pixmap.scaled(self.image_label1.width(),
                               self.image_label1.height())
        self.image_label1.setPixmap(pixmap)
        self.imgAs2DArray = cv2.imread(name[0])

        self.btn_prep.setEnabled(True)

    def preprocessing(self):
        b, g, r = cv2.split(self.imgAs2DArray)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_green = self.clahe.apply(g)

        img = toimage(contrast_green)
        qim = ImageQt(img)
        pixMap = QtGui.QPixmap.fromImage(qim)

        pixmap = pixMap.scaled(self.image_label1.width(),
                               self.image_label1.height())

        self.image_label1.setPixmap(pixmap)

        self.imgAs2DArray = contrast_green
        self.btn_prep.setEnabled(False)
        self.btn_start.setEnabled(True)

    def start(self):
        r1 = cv2.morphologyEx(self.imgAs2DArray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                              iterations=1)
        R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
        R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
        f4 = cv2.subtract(R3, self.imgAs2DArray)
        self.f5 = self.clahe.apply(f4)

        ret, f6 = cv2.threshold(self.f5, 15, 255, cv2.THRESH_BINARY)         #RET NIE UŻYWAMY

        f6 = cv2.bitwise_not(f6)        #ODWRACAMY KOLORY

        img = toimage(f6)
        qim = ImageQt(img)
        pixMap = QtGui.QPixmap.fromImage(qim)

        pixmap = pixMap.scaled(self.image_label2.width(),
                               self.image_label2.height())

        self.image_label2.setPixmap(pixmap)

        self.imgAs2DArray = f6
        self.btn_start.setEnabled(False)
        self.btn_post.setEnabled(True)

    def postprocessing(self):

        mask = np.ones(self.f5.shape[:2], dtype="uint8") * 255
        contours, hierarchy = cv2.findContours(self.imgAs2DArray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) <= 200:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        im = cv2.bitwise_and(self.f5, self.f5, mask=mask)
        ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
        newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

        fundus_eroded = cv2.bitwise_not(newfin)
        xmask = np.ones(self.img.shape[:2], dtype="uint8") * 255
        xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in xcontours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
            if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
                shape = "circle"
            else:
                shape = "veins"
            if (shape == "circle"):
                cv2.drawContours(xmask, [cnt], -1, 0, -1)

        finimage = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
        blood_vessels = cv2.bitwise_not(finimage)

        imgr = toimage(blood_vessels)
        qim = ImageQt(imgr)
        pixMap = QtGui.QPixmap.fromImage(qim)

        pixmap = pixMap.scaled(self.image_label3.width(),
                               self.image_label3.height())

        self.image_label3.setPixmap(pixmap)

        self.imgAs2DArray = blood_vessels
        self.btn_post.setEnabled(False)
        QtGui.QGuiApplication.processEvents()
        self.wyliczMiary()

    def wyliczMiary(self):

        self.label.setText("TRWA WYLICZANIE MIAR")
        QtGui.QGuiApplication.processEvents()

        TP = 0
        FP = 0
        FN = 0
        TN = 0

        img = self.imgAs2DArray
        img_master = cv2.imread(self.path + "/" +self.img_master_name)
        img_master = cv2.bitwise_not(img_master)                        # 1 - nie ma naczynia
                                                                        # 0 - jest naczynko
        zeros = np.array([0, 0, 0])
        ones = np.array([255, 255, 255])

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (img_master[i][j] == zeros).all():       #JEŻELI JEST NACZYNIE W TYM MIEJSCU
                    if(img[i][j] == 0):                                                             #I NASZ WYNIK TO POTWIERDZA
                        TP += 1
                    elif(img[i][j] == 255):                                                         #A NASZ WYNIK MÓWI ŻE JEST INACZEJ
                        FN += 1
                elif (img_master[i][j] == ones).all():      #JEŻELI NIE MA NACZYNKA W TYM MIEJSCU
                    if (img[i][j] == 255):                                                            # I NASZ WYNIK TO POTWIERDZA
                        TN += 1
                    elif (img[i][j] == 0):                                                        # A NASZ WYNIK MÓWI ŻE JEST INACZEJ
                        FP += 1

        print("TP: " + TP.__str__())
        print("FN: " + FN.__str__())
        print("TN: " + TN.__str__())
        print("FP: " + FP.__str__())

        print("PRECYZJA(PRECISSION): " + ((TP)/(TP+FP)).__str__())
        print("CZUŁOŚĆ(RECALL,SENSITIVITY): " + ((TP)/(TP+FN)).__str__())
        print("TRAFNOŚĆ(ACCURACY): " + ((TP + TN)/(TP+TN+FP+FN)).__str__())
        print("SWIOSTOŚĆ(SPECIFICITY): " + ((TN)/(TN+FP)).__str__())
        print("ŚREDNIA CZUŁ. I SWOIST.: "+(( ((TP)/(TP+FN)) + ((TN)/(TN+FP)) )/2 ).__str__())

        self.label1_miara_wart.setText(round(((TP)/(TP+FP)),3).__str__())
        self.label2_miara_wart.setText(round(((TP)/(TP+FN)),3).__str__())
        self.label3_miara_wart.setText(round(((TP + TN)/(TP+TN+FP+FN)),3).__str__())
        self.label4_miara_wart.setText(round(((TN)/(TN+FP)),3).__str__())
        self.label5_miara_wart.setText(round((( ((TP)/(TP+FN)) + ((TN)/(TN+FP)) )/2 ),3).__str__())

        self.label.setText("MIARY")
        QtGui.QGuiApplication.processEvents()


def run():
    app = QtWidgets.QApplication(sys.argv)
    GUI = Window()
    app.exec_()


run()






