from PySide6.QtGui import QImage
from PySide6.QtCore import QThread, Signal
import cv2
import time

class CameraThread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, camera_num, parent=None):
        QThread.__init__(self, parent)
        self.camera_num = camera_num
        self.stop = True
        self.save_frame = False
        self.save_path = ""
    
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_num)
        while True:
            if (not self.stop) and self.cap.isOpened():
                ret, self.frame = self.cap.read()
                if ret:
                    color_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    img = QImage(color_frame.data, color_frame.shape[1], color_frame.shape[0], QImage.Format_RGB888)
                    self.updateFrame.emit(img)
            
            elif self.save_frame:
                cv2.imwrite(self.save_path, self.frame)
                self.save_frame = False

            else:
                time.sleep(1)


    def stop_thread(self):
        self.cap.release()
        self.terminate()


