from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import tensorflow as tf
from main_gui import Ui_ContainerHome
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import sys



class main_app(QWidget, Ui_ContainerHome):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.main_page.show()
        self.btn_home_img.clicked.connect(self.init_imagepage)
        self.btn_home_cam.clicked.connect(self.webcamview)
        

    def ImageUpdateSlot(self, Image):
        self.windowslabel.setPixmap(QPixmap.fromImage(Image))
        
        
    def CancelFeed(self):
        self.cam_record.stop()
        
    def webcamview(self):
        self.stop_btn.clicked.connect(self.CancelFeed)
        self.cam_record = cam_record()
        self.cam_record.start()
        self.cam_record.ImageUpdate.connect(self.ImageUpdateSlot)
        
    def init_imagepage(self):
        self.main_page.hide()
        self.img_page.show()

        
        
class cam_record(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    
    
    def stop(self):
        self.ThreadActive = False
        self.quit()

    
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    welcome = main_app()
    welcome.show()

    app.exec()
