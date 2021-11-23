from modules.predict import frame_predict
import numpy as np
import cv2
#import os
import cvlib as cv
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
            # apply face detection
            face, confidence = cv.detect_face(frame)
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # loop through detected faces
                for idx, f in enumerate(face):
                    # get corner points of face rectangle        
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]

                    # draw rectangle over face
                    cv2.rectangle(Image, (startX,startY), (endX,endY), (0,255,0), 2)

                    # crop the detected face region
                    face_crop = np.copy(Image[startY:endY,startX:endX])
                    
                    if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                        continue

                    # get predictions

                    label = frame_predict(face_crop)

                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    # write label and confidence above face rectangle
                    cv2.putText(Image, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                
                
 
                #FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    
        Capture.release()
        cv2.destroyAllWindows()
        
    def stop(self):
        self.ThreadActive = False
        self.quit()

    
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    welcome = main_app()
    welcome.show()

    app.exec()
