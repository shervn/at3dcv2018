import sys
from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QMessageBox
from PyQt5.QtGui import QFont

from open3d import *

from Augmentation.augmentation import Augmentor
#from Segmentation.segmentation import Segmenter
from Reconstruction.reconstruction import Reconstructor

sys.path.append("Reconstruction/ReconstructionSystem/sensors")
from record import record
import json
from realsense_recorder import realsense_recorder
from os import makedirs
from os.path import exists, join
import shutil


class View(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        # button to record
        record_btn = QPushButton('Record', self)
        record_btn.setToolTip('Press to record with your camera')
        record_btn.clicked.connect(self.Record)
        record_btn.resize(record_btn.sizeHint())
        record_btn.move(50, 50)


        # 3d reconstruction
        reconstruct_btn = QPushButton('Reconstruct', self)
        reconstruct_btn.clicked.connect(self.Reconstruct)
        reconstruct_btn.resize(reconstruct_btn.sizeHint())
        reconstruct_btn.move(50, 100)

        # Augmentation
        augment_btn = QPushButton('Augment', self)
        augment_btn.clicked.connect(self.Augment)
        augment_btn.resize(reconstruct_btn.sizeHint())
        augment_btn.move(200, 50)

        # set geometry
        self.setGeometry(200, 100, 750, 500)
        self.setWindowTitle('App')
        self.show()

    def Record(self):

        config = "Reconstruction/ReconstructionSystem/config/realsense.json"
        if config is not None:
            with open(config) as json_file:
                config = json.load(json_file)

        output_folder = config['path_dataset']

        path_output = output_folder
        path_depth = join(output_folder, "depth")
        path_color = join(output_folder, "color")

        self.make_clean_folder(path_output, path_depth, path_color)


    def make_clean_folder(self, path_folder, path_depth, path_color):

        if not exists(path_folder):
            makedirs(path_folder)
            makedirs(path_depth)
            makedirs(path_color)
        else:
            choice = QMessageBox.question(self, 'Message', "Do you want to overwrite the previous data?",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if choice == QMessageBox.Yes:
                shutil.rmtree(path_folder)
                makedirs(path_folder)
                makedirs(path_depth)
                makedirs(path_color)
                QMessageBox.Close
                realsense_recorder(path_folder)
            else:
                pass


    def Reconstruct(self):
        self.r = Reconstructor()

    def Segment(self):
        self.s = Segmenter(self.r.reconstructed_pointcloud)

    def Augment(self):
        name = "/home/pti/Downloads/tum/at3dcv/project/pointclouds_for_fun/scene0000_00_vh_clean_2.labels.ply"
        dummy_pcl = read_point_cloud(name)
        # draw_geometries([self.r])
        self.t = Augmentor(self, dummy_pcl)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = View()
    sys.exit(app.exec_())

    # r = Reconstructor()
    # s = Segmantor(r.reconstructed_pointcloud)
    # t = Augmentor(s.labled_pointcloud)
