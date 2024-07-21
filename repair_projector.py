# This Python file uses the following encoding: utf-8
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

from pathlib import Path
import sys
import argparse
import cv2
import subprocess
import json
import time
import numpy as np
import shutil

from utils.file_ops import list_images, load_image
from utils.cam_ops import CameraThread

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QFile, Qt
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QThread, Signal


class RepairProjector(QWidget):
    def __init__(self, args_parser, parent=None):
        super(RepairProjector, self).__init__(parent)
        self.load_ui(args_parser)


    def load_ui(self, args_parser):
        loader = QUiLoader()
        path = Path(__file__).resolve().parent / "form.ui"
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)

        if os.path.exists(args_parser.config):
            with open(args_parser.config) as json_file:
                self.config_dict = json.load(json_file)

        self.repair_im_list, self.damage_im_list = list_images(self.config_dict["datapath"])
        self.damage_repair_temp_dir = os.path.join(
            self.config_dict["datapath"], self.config_dict["damage_repair_path"])
        if not os.path.exists(self.damage_repair_temp_dir):
            os.makedirs(self.damage_repair_temp_dir)

        for dim in self.damage_im_list:
            self.ui.listWidget_damage.addItem(dim.name)

        for rim in self.repair_im_list:
            self.ui.listWidget_repair.addItem(rim.name)

        self.ui.listWidget_repair.currentItemChanged.connect(self.select_repair_image)
        self.ui.listWidget_damage.currentItemChanged.connect(self.select_damage_image)
        self.ui.pushButton_webcam.pressed.connect(self.toggle_webcam)
        
        self.ui.curr_fab_type = "damage"
        self.ui.comboBox_fab_type.currentTextChanged.connect(self.update_fab_type)
        
        self.live_capture_flag = False
        self.cam = CameraThread(self.config_dict["camera_number"])
        self.cam.updateFrame.connect(self.update_image)
        self.cam.start()

        self.process_thread = ProcessThread(self.config_dict)
        self.process_thread.updateResult.connect(self.show_result)
        self.process_thread.start()

        self.ui.pushButton_save.pressed.connect(self.save_captured_image)
        self.ui.pushButton_preview.pressed.connect(self.preview_repaired)

        ui_file.close()


    def update_fab_type(self, value):
        if value == 0:
            self.ui.curr_fab_type = "damage"
        else:
            self.ui.curr_fab_type = "repair"


    def select_damage_image(self):
        indx = self.ui.listWidget_damage.currentRow()
        self.curr_damage_im_path = self.damage_im_list[indx]

        self.damage_img = load_image(str(self.curr_damage_im_path))
        self.damage_img = cv2.cvtColor(self.damage_img, cv2.COLOR_BGR2RGB)
        h, w, ch = self.damage_img.shape
        bytesPerLine = ch * w
        qimg = QImage(self.damage_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        qimg = qimg.scaled(640, 480, Qt.KeepAspectRatio)
        self.damage_pixmap = QPixmap.fromImage(qimg)
        self.ui.label_damage.setPixmap(self.damage_pixmap)


    def select_repair_image(self):
        indx = self.ui.listWidget_repair.currentRow()
        self.curr_repair_im_path = self.repair_im_list[indx]

        self.repair_img = load_image(str(self.curr_repair_im_path))
        self.repair_img = cv2.cvtColor(self.repair_img, cv2.COLOR_BGR2RGB)
        h, w, ch = self.repair_img.shape
        bytesPerLine = ch * w
        qimg = QImage(self.repair_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        qimg = qimg.scaled(640, 480, Qt.KeepAspectRatio)
        self.repair_pixmap = QPixmap.fromImage(qimg)
        self.ui.label_repair.setPixmap(self.repair_pixmap)


    def toggle_webcam(self):
        if not self.live_capture_flag:
            self.live_capture_flag = True
            self.cam.stop = False
            self.ui.pushButton_webcam.setText("Stop Webcam")
        else:
            self.live_capture_flag = False
            self.cam.stop = True
            self.ui.pushButton_webcam.setText("Start Webcam")


    def update_image(self, qimg):
        qimg = qimg.scaled(640, 480, Qt.KeepAspectRatio)
        if self.ui.curr_fab_type == "damage":
            self.damage_pixmap = QPixmap.fromImage(qimg)
            self.ui.label_damage.setPixmap(self.damage_pixmap)
        else:
            self.repair_pixmap = QPixmap.fromImage(qimg)
            self.ui.label_repair.setPixmap(self.repair_pixmap)


    def save_captured_image(self):
        self.cam.save_frame = True
        img_name = self.ui.lineEdit_img_name.text()
        if img_name != "":
            img_name = img_name + ".png"
        if self.ui.curr_fab_type == "damage":
            if img_name == "":
                img_name = "damage.png"
            self.cam.save_path = os.path.join(self.config_dict["damage_path"], img_name)
        else:
            if img_name == "":
                img_name = "repair.png"
            self.cam.save_path = os.path.join(self.config_dict["repair_path"], img_name)

        self.repair_im_list, self.damage_im_list = list_images(self.config_dict["datapath"])

        self.ui.listWidget_damage.clear()
        self.ui.listWidget_repair.clear()

        for dim in self.damage_im_list:
            self.ui.listWidget_damage.addItem(dim.name)

        for rim in self.repair_im_list:
            self.ui.listWidget_repair.addItem(rim.name)



    def show_result(self, flag):
        print("image will be showed now")

        self.preview_img = load_image(self.curr_preview_path)
        org_damage_img = cv2.imread(str(self.curr_damage_im_path))
        mask_file = os.path.join(self.config_dict["damage_mask_path"], str(self.curr_damage_im_path.name.replace(".jpg", ".png")))
        mask_img = cv2.imread(mask_file, 0)
        mask_img = mask_img.astype(np.float32)
        mask_img = cv2.GaussianBlur(mask_img, (5,5), sigmaX=5, sigmaY=5)
        
        self.preview_img[mask_img < 0.5] = org_damage_img[mask_img < 0.5]

        self.preview_img = cv2.cvtColor(self.preview_img, cv2.COLOR_BGR2RGB)
        h, w, ch = self.preview_img.shape
        bytesPerLine = ch * w
        qimg = QImage(self.preview_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        qimg = qimg.scaled(640, 480, Qt.KeepAspectRatio)
        self.preview_pixmap = QPixmap.fromImage(qimg)
        self.ui.label_preview.setPixmap(self.preview_pixmap)

        if os.path.exists(self.curr_damage_im_path_for_repair):
            os.remove(self.curr_damage_im_path_for_repair)


    def preview_repaired(self):

        # Copy damage image to an image with name that includes repair style image
        _, damage_ext = os.path.splitext(str(self.curr_damage_im_path))
        _, repair_patch_ext = os.path.splitext(str(self.curr_repair_im_path))
        _, repair_patch_ext = os.path.splitext(str(self.curr_repair_im_path))
        self.repair_base_name = os.path.basename(self.curr_repair_im_path.name).replace(repair_patch_ext, "")
        
        self.curr_damage_im_path_for_repair = os.path.join(self.damage_repair_temp_dir, self.curr_damage_im_path.name.replace(
            damage_ext, "_" + self.repair_base_name + damage_ext))
        # print(self.curr_damage_im_path)
        # print(self.curr_damage_im_path_for_repair)
        shutil.copy(self.curr_damage_im_path, self.curr_damage_im_path_for_repair)

        # tmp_damage_image = cv2.imread(self.curr_damage_im_path_for_repair)
        # mask_file = os.path.join(self.config_dict["damage_mask_path"], str(
        #     self.curr_damage_im_path.name.replace(".jpg", ".png")))
        # mask_img = cv2.imread(mask_file, 0)
        # mask_img = mask_img.astype(np.float32)
        # mask_img = cv2.GaussianBlur(mask_img, (5, 5), sigmaX=5, sigmaY=5)
        # tmp_damage_image[mask_img > 0.5] = 255
        # cv2.imwrite(self.curr_damage_im_path_for_repair, tmp_damage_image)

        preview_base_name = os.path.basename(self.curr_damage_im_path_for_repair)
        self.curr_preview_path = os.path.join(
            self.config_dict["preview_path"], preview_base_name.replace(".png", ".jpg"))
        self.curr_preview_path = self.curr_preview_path.replace(".jpg", "_mask_generated.jpg")

        if not os.path.exists(self.curr_preview_path):
            self.process_thread.content_im = str(self.curr_damage_im_path_for_repair)
            self.process_thread.content_im_mask = os.path.join(
                self.config_dict["damage_mask_path"], str(self.curr_damage_im_path.name.replace(".jpg", ".png")))
            self.process_thread.style_im = str(self.curr_repair_im_path)
            self.process_thread.run_job = True
            self.ui.label_preview.setText("Please wait while the image is synthesized")

        else:
            self.preview_img = load_image(self.curr_preview_path)
            org_damage_img = cv2.imread(str(self.curr_damage_im_path))
            mask_file = os.path.join(self.config_dict["damage_mask_path"], str(self.curr_damage_im_path.name.replace(".jpg", ".png")))
            mask_img = cv2.imread(mask_file, 0)
            mask_img = mask_img.astype(np.float32)
            mask_img = cv2.GaussianBlur(mask_img, (5,5), sigmaX=5, sigmaY=5)
            
            self.preview_img[mask_img < 0.5] = org_damage_img[mask_img < 0.5]

            self.preview_img = cv2.cvtColor(self.preview_img, cv2.COLOR_BGR2RGB)
            h, w, ch = self.preview_img.shape
            bytesPerLine = ch * w
            qimg = QImage(self.preview_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
            qimg = qimg.scaled(640, 480, Qt.KeepAspectRatio)
            self.preview_pixmap = QPixmap.fromImage(qimg)
            self.ui.label_preview.setPixmap(self.preview_pixmap)

            if os.path.exists(self.curr_damage_im_path_for_repair):
                os.remove(self.curr_damage_im_path_for_repair)

    def close(self) -> bool:
        self.cam.stop = True
        self.cam.stop_thread()
        return super().close()


class ProcessThread(QThread):
    updateResult = Signal(bool)

    def __init__(self, config_dict, parent=None):
        QThread.__init__(self, parent)
        self.config_dict = config_dict
        self.content_im = "images/damages/W1P1D3.png"
        self.content_im_mask = "images/damage_masks/W1P1D3.png"
        self.style_im = "images/repairs/repair_03.jpg"
        self.run_job = False

    def run(self):
        while True:
            if self.run_job:
                str_command = ["python", "StyleTransfer/tools/test.py", "--config-file", "StyleTransfer/configs/wct_test.yaml", 
                    "--content " , self.content_im, "--style ",  self.content_im + "," + self.style_im, "--mask ", self.content_im_mask]

                print("running the command: ", str_command)
                subprocess.run(" ".join(str_command),  shell=True, check=True)

                self.run_job = False
                self.updateResult.emit(True)

            else:
                time.sleep(1)

    def stop_thread(self):
        self.terminate()






def main(app, args_parser):
    widget = RepairProjector(args_parser)
    widget.show()
    sys.exit(app.exec())



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', type=str, dest='config', help='configuration file')
    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    # Create the application instance.
    app = QApplication([])

    main(app, args_parser)
