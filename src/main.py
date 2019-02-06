import sys

from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QVBoxLayout, \
                            QMainWindow, QApplication, QHBoxLayout, QGridLayout, QFrame, QAbstractButton, QLabel, \
                            QComboBox
from PyQt5.QtGui import QFont, QImage, QPainter, QIcon, QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QCoreApplication, Qt, QObject, QFile, QTextStream, QSize

from open3d import *
from Reconstruction.reconstruction import Reconstructor
from Segmentation.segmentation import Segmenter
from Augmentation.augmentation import Augmentor
from Augmentation.augmentation_ui import AugmentationUI

from paths import *
from Augmentation.config import *

import json
from os import makedirs
from os.path import exists, join
import shutil
from pyqtgraph import ImageView
import cv2

sys.path.append(macpyrealsense2)
import pyrealsense2 as rs

import numpy as np
import argparse
from enum import IntEnum

sys.path.append(images_path)
sys.path.append(layout_path)
import breeze_resources



try:
    # Python 2 compatible
    input = raw_input
except NameError:
    pass

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


class PicButton(QAbstractButton):
    def __init__(self, pixmap, pixmap_hover, pixmap_pressed, parent=None):
        super(PicButton, self).__init__(parent)
        self.pixmap = pixmap
        self.pixmap_hover = pixmap_hover
        self.pixmap_pressed = pixmap_pressed
        self.setDown(False)

        self.pressed.connect(self.update)
        self.released.connect(self.update)

    def paintEvent(self, event):
        pix = self.pixmap_hover if self.underMouse() else self.pixmap

        if self.isDown():
            temp = self.pixmap
            self.pixmap = self.pixmap_pressed

            pix = self.pixmap_pressed
            # self.pixmap_pressed = temp

        painter = QPainter(self)
        painter.drawPixmap(event.rect(), pix)

    def enterEvent(self, event):
        self.update()

    def leaveEvent(self, event):
        self.update()

    def sizeHint(self):
        return QSize(200, 200)

class View(QWidget):

    sgnStop = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._thread = None

        self.initUI()

    def initUI(self):

        central_widget = QWidget()
        layout = QGridLayout(central_widget)
        layout.setSpacing(10)

        # self.setCentralWidget(central_widget)

        self._thread = None
        if self._thread == None:
            self._thread = QThread()

        self._thread.start()
        self.vid = ShowVideo()
        self.vid.moveToThread(self._thread)
        image_viewer = ImageViewer()

        self.vid.VideoSignal.connect(image_viewer.setImage)
        self.sgnStop.connect(self.vid.stopVideo)


        # frame = QFrame()
        # frame.resize(300,300)
        # frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        # frame.setStyleSheet("background-color: rgb(255, 255, 255)")


        # dummy button
        dummy_button = QPushButton('dummy')
        dummy_button.clicked.connect(self.vid.startVideo)
        dummy_button.setFixedWidth(100)
        dummy_button.setFixedHeight(100)


        # Button to start the videocapture:
        # start_button = QPushButton('Start')
        # start_button = QPushButton()
        # start_button.setIcon(QIcon(QPixmap("play.svg")))
        start_button = PicButton(QPixmap(join(images_path + "/play.png")), QPixmap(join(images_path + "/play.png")), QPixmap(join(images_path + "/play.png")))
        # start_button.setIcon(QIcon('play.png'))
        # start_button.setIconSize(QSize(24, 24))
        start_button.clicked.connect(self.vid.startVideo)
        start_button.setFixedWidth(50)
        start_button.setFixedHeight(50)


        # stop_button = QPushButton('Stop')
        stop_button = PicButton(QPixmap(join(images_path + "/rec.png")), QPixmap(join(images_path + "/rec.png")), QPixmap(join(images_path + "/rec.png")))

        stop_button.clicked.connect(self.disable)
        stop_button.setFixedWidth(45)
        stop_button.setFixedHeight(45)

        # void QGridLayout::addWidget(QWidget * widget, int fromRow, int fromColumn, int rowSpan, int columnSpan, Qt::Alignment alignment = 0)

        # layout.setContentsMargins(left, top, right, bottom)
        layout.setContentsMargins(100, 100, 100, 100)
        image_viewer.setFixedWidth(720)
        image_viewer.setFixedHeight(290)


        hbox_image = QHBoxLayout()
        hbox_image.addWidget(image_viewer)
        layout.addLayout(hbox_image, 0, 0, 1, 1, Qt.AlignLeft)

        # label = QLabel(self)
        # label.setStyleSheet("background-color: white; inset grey; min-height: 200px;")
        # label.setFrameShape(QFrame.Panel)
        # label.setFrameShadow(QFrame.Sunken)
        # label.setLineWidth(3)
        # label.setFixedWidth(100)
        # label.setFixedHeight(200)


        frame_rate = QLabel("Select frame rate", self)
        comboBox = QComboBox(self)
        comboBox.addItem("6")
        comboBox.addItem("15")
        comboBox.addItem("30")
        comboBox.setCurrentIndex(1)
        comboBox.activated[str].connect(self.style_choice)


        hbox_dummy = QVBoxLayout()
        hbox_dummy.addWidget(frame_rate)
        hbox_dummy.addWidget(comboBox)
        layout.addLayout(hbox_dummy, 0, 1, 1, 1, Qt.AlignCenter)

        hbox_start_buttons = QHBoxLayout()
        hbox_start_buttons.addWidget(start_button)
        hbox_start_buttons.addWidget(stop_button)
        layout.addLayout(hbox_start_buttons, 0, 0, 1, 1, Qt.AlignBottom| Qt.AlignCenter)
        # layout.addLayout(hbox_start_buttons, 1, 0, 1, 1, Qt.AlignTop| Qt.AlignCenter)

        # layout.setVerticalSpacing(30)
        layout.setVerticalSpacing(100)

        # 3d reconstruction
        reconstruct_btn = QPushButton()
        reconstruct_btn.clicked.connect(self.Reconstruct)

        reconstruct_btn.setIcon(QIcon(join(images_path + '/construct.png')))
        reconstruct_btn.setIconSize(QSize(70, 70))
        # reconstruct_btn.resize(reconstruct_btn.sizeHint())
        reconstruct_btn.setFixedWidth(100)
        reconstruct_btn.setFixedHeight(90)
        # layout.addWidget(reconstruct_btn, 4, 0)
        reconstruct_label = QLabel("  Reconstruct", self)

        # View pointclouds
        view_btn = QPushButton()
        view_btn.clicked.connect(self.Augment)
        # show_btn.resize(reconstruct_btn.sizeHint())
        view_btn.setIcon(QIcon(join(images_path + '/view.png')))
        view_btn.setIconSize(QSize(50, 50))
        view_btn.setFixedWidth(100)
        view_btn.setFixedHeight(90)
        view_label = QLabel("         View", self)


        # Segment
        segment_btn = QPushButton()
        segment_btn.clicked.connect(self.Augment)
        # compare_btn.resize(reconstruct_btn.sizeHint())
        segment_btn.setIcon(QIcon(join(images_path + '/seg3.png')))
        segment_btn.setIconSize(QSize(80, 80))
        segment_btn.setFixedWidth(100)
        segment_btn.setFixedHeight(90)
        segmentation_label = QLabel("     Segment", self)


        # View pointclouds
        show_btn = QPushButton()
        show_btn.clicked.connect(self.Augment)
        # show_btn.resize(reconstruct_btn.sizeHint())
        show_btn.setIcon(QIcon(join(images_path + '/view1.png')))
        show_btn.setIconSize(QSize(50, 50))
        show_btn.setFixedWidth(100)
        show_btn.setFixedHeight(90)
        show_label = QLabel("         Show", self)

        # Augmentation
        augment_btn = QPushButton()
        augment_btn.clicked.connect(self.Augment)
        # augment_btn.resize(reconstruct_btn.sizeHint())
        augment_btn.setIcon(QIcon(join(images_path + '/magic.png')))
        augment_btn.setIconSize(QSize(70, 70))
        augment_btn.setFixedWidth(100)
        augment_btn.setFixedHeight(90)
        # layout.addWidget(augment_btn, 4, 2)
        augmentation_label = QLabel("        Magic", self)



        hbox_buttons = QHBoxLayout()
        vbox_rec = QVBoxLayout()
        vbox_rec.addWidget(reconstruct_btn)
        vbox_rec.addWidget(reconstruct_label)
        hbox_buttons.addLayout(vbox_rec)

        vbox_view = QVBoxLayout()
        vbox_view.addWidget(view_btn)
        vbox_view.addWidget(view_label)
        # vbox_aug.setAlignment(Qt.AlignHCenter)
        hbox_buttons.addLayout(vbox_view)

        vbox_comp = QVBoxLayout()
        vbox_comp.addWidget(segment_btn)
        vbox_comp.addWidget(segmentation_label)
        hbox_buttons.addLayout(vbox_comp)

        vbox_show = QVBoxLayout()
        vbox_show.addWidget(show_btn)
        vbox_show.addWidget(show_label)
        hbox_buttons.addLayout(vbox_show)

        vbox_aug = QVBoxLayout()
        vbox_aug.addWidget(augment_btn)
        vbox_aug.addWidget(augmentation_label)
        # vbox_aug.setAlignment(Qt.AlignHCenter)
        hbox_buttons.addLayout(vbox_aug)


        # hbox_buttons.addWidget(compare_btn)
        # hbox_buttons.addWidget(show_btn)
        # hbox_buttons.addWidget(augment_btn)
        layout.addLayout(hbox_buttons, 3, 0, 1, 2)

        # hbox_buttons1 = QHBoxLayout()
        # hbox_buttons1.addWidget(reconstruct_label)
        # hbox_buttons1.addWidget(reconstruct_label)
        # hbox_buttons1.addWidget(reconstruct_label)
        # hbox_buttons1.addWidget(reconstruct_label)
        # layout.addLayout(hbox_buttons1, 4, 0)

        # hbox1 = QHBoxLayout()
        # vlayout.addStretch(1)
        # layout.addWidget(start_button, 2, 0)
        # layout.addWidget(stop_button, 2, 1)
        # vlayout.setAlignment(Qt.AlignRight)
        # hbox1.addStretch(1)
        # layout.setAlignment(Qt.AlignTop)
        # hbox1.addStretch(-1)
        # vlayout.addLayout(hbox1)
        # layout.setRowStretch(3, 1)
        # layout.setColumnStretch(0, 1)

        # # layout.setColumnStretch(1, 2)
        # # layout.setRowStretch(2, 5)
        # layout.addWidget(image_viewer, 0, 0, 1, 4)
        #
        # # layout.addStretch()
        # layout.addWidget(start_button, 2, 0)
        # layout.addWidget(stop_button, 2, 1)


        # # button to record
        # record_btn = QPushButton('Record', self)
        # record_btn.setToolTip('Press to record with your camera')
        # record_btn.clicked.connect(self.Record)
        # record_btn.resize(record_btn.sizeHint())
        # layout.addWidget(record_btn)



        self.setLayout(layout)
        # set geometry
        self.setGeometry(0, 0, 1000, 650)
        self.setWindowTitle('App')
        self.show()

    def style_choice(self, text):
        # self.style_choice.setText(text)
        self.vid.frame_rate = int(text)
        print(int(self.vid.frame_rate))



    def disable(self):
        print ('sending stop signal to the worker object')
        self.sgnStop.emit()  # send a queuedconnection type signal to the worker, because its in another thread

    def Record(self):

        config = camera_config_path

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
                self.realsense_recorder(path_folder)
            else:
                pass


    def Reconstruct(self):
        self.r = Reconstructor()

    def Segment(self):
        self.s = Segmenter(self.r.reconstructed_pointcloud)

    def Augment(self):
        
        name = reconstructed_scene
        dummy_pcl = read_point_cloud(name)

        name_labels = segmented_reconstructed_scene
        dummy_pcl_labels = read_point_cloud(name_labels)
        
        t = Augmentor(dummy_pcl, dummy_pcl_labels)
        self.augmentation_ui = AugmentationUI(t)




class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QImage()
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QImage()
        # self.image.scaledToWidth(854)


    @pyqtSlot(QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


class ShowVideo(QObject):
    VideoSignal = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)
        self.frame_rate = 15
        print("ShowVideo")

    @pyqtSlot()
    def startVideo(self):

        print("start videoooooooooooooooooo")
        self._running = True
        config = camera_config_path
        if config is not None:
            with open(config) as json_file:
                config = json.load(json_file)

        output_folder = config['path_dataset']

        path_depth = join(output_folder, "depth")
        path_color = join(output_folder, "color")

        self.make_clean_folder(output_folder)
        self.make_clean_folder(path_depth)
        self.make_clean_folder(path_color)

        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        print("frame_rate")
        print(self.frame_rate)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.frame_rate)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.frame_rate)

        # Start streaming
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()

        # Using preset HighAccuracy for recording

        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_scale = depth_sensor.get_depth_scale()

        # We will not display the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 3  # 3 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Streaming loop
        frame_count = 0
        try:
            while self._running:
                QCoreApplication.processEvents()
                # Get frameset of color and depth
                frames = pipeline.wait_for_frames()

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                if frame_count == 0:
                    self.save_intrinsic_as_json(join(output_folder,
                                                     "camera_intrinsic.json"), color_frame)
                cv2.imwrite("%s/%06d.png" % \
                            (path_depth, frame_count), depth_image)
                cv2.imwrite("%s/%06d.jpg" % \
                            (path_color, frame_count), color_image)
                print("Saved color + depth image %06d" % frame_count)
                frame_count += 1

                # Remove background - Set pixels further than clipping_distance to grey
                grey_color = 153
                # depth image is 1 channel, color is 3 channels
                depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
                bg_removed = np.where((depth_image_3d > clipping_distance) | \
                                      (depth_image_3d <= 0), grey_color, color_image)

                # Render images
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.09),
                    cv2.COLORMAP_JET)
                images = np.hstack((bg_removed, depth_colormap))
                color_swapped_image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                height, width, _ = images.shape
                print(height, width)

                qt_image = QImage(color_swapped_image,
                                        width,
                                        height,
                                        color_swapped_image.strides[0],
                                        QImage.Format_RGB888).scaledToWidth(720)


                # cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('Recorder Realsense', images)
                key = cv2.waitKey(1)

                # if 'esc' button pressed, escape loop and exit program
                if key == 27:
                    cv2.destroyAllWindows()
                    break
                self.VideoSignal.emit(qt_image)
        finally:
            pipeline.stop()


    def make_clean_folder(self, path_folder):
        if not exists(path_folder):
            makedirs(path_folder)
        else:

            shutil.rmtree(path_folder)
            makedirs(path_folder)


    def save_intrinsic_as_json(self, filename, frame):
        intrinsics = frame.profile.as_video_stream_profile().intrinsics
        with open(filename, 'w') as outfile:
            obj = json.dump({'width': intrinsics.width,
                             'height': intrinsics.height,
                             'intrinsic_matrix':
                                 [intrinsics.fx, 0, 0,
                                  0, intrinsics.fy, 0,
                                  intrinsics.ppx, intrinsics.ppy, 1]},
                            outfile, indent=4)

    @pyqtSlot()
    def stopVideo(self):
        print ('stop signal received, switching while loop condition to false')
        self._running = False


if __name__ == '__main__':

    app = QApplication(sys.argv)
    file = QFile(":/dark.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())
    view = View()
    sys.exit(app.exec_())