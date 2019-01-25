import sys, os
from PyQt5.QtWidgets import (QWidget, QToolTip,
    QPushButton, QApplication)
from PyQt5.QtGui import QFont

#from Augmentation.augmentation import Augmentor
#from Segmentation.segmentation import Segmenter
from Reconstruction.reconstruction import Reconstructor

rel_path = os.path.realpath('')
sys.path.append(rel_path + '/src/Reconstruction/ReconstructionSystem/sensors')
from record import record


class View(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        # button to record
        record_btn = QPushButton('Record', self)
        record_btn.setToolTip('Press to record with your camera')
        record_btn.clicked.connect(record)
        record_btn.resize(record_btn.sizeHint())
        record_btn.move(50, 50)


        # 3d reconstruction
        reconstruct_btn = QPushButton('Reconstruct', self)
        reconstruct_btn.clicked.connect(self.Reconstruct)
        reconstruct_btn.resize(reconstruct_btn.sizeHint())
        reconstruct_btn.move(50, 100)

        # set geometry
        self.setGeometry(200, 100, 750, 500)
        self.setWindowTitle('Quit button')
        self.show()

    def Reconstruct(self):
        self.r = Reconstructor()

    def Segment(self):
        self.s = Segmenter(self.r.reconstructed_pointcloud)

    def Augmentor(self):
        self.t = Augmentor(self.s.labled_pointcloud)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = View()
    sys.exit(app.exec_())

    # r = Reconstructor()
    # s = Segmantor(r.reconstructed_pointcloud)
    # t = Augmentor(s.labled_pointcloud)
