import sys

from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QVBoxLayout, \
                            QMainWindow, QApplication, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QFont, QImage, QPainter
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QCoreApplication, Qt, QObject

from open3d import *

from src.Reconstruction.reconstruction import Reconstructor
from src.Segmentation.segmentation import Segmenter
from src.Augmentation.augmentation import Augmentor

# from paths import sensors

import json
from os import makedirs
from os.path import exists, join
import shutil
from pyqtgraph import ImageView
import cv2
import pyrealsense2 as rs
import numpy as np
import argparse
from enum import IntEnum

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


class View(QMainWindow):

    sgnStop = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._thread = None
        self.initUI()

    def initUI(self):

        central_widget = QWidget()
        layout = QGridLayout(central_widget)

        self.setCentralWidget(central_widget)



        self._thread = None
        if self._thread == None:
            self._thread = QThread()

        self._thread.start()
        self.vid = ShowVideo()
        self.vid.moveToThread(self._thread)
        image_viewer = ImageViewer()

        self.vid.VideoSignal.connect(image_viewer.setImage)
        self.sgnStop.connect(self.vid.stopVideo)




        # Button to start the videocapture:
        start_button = QPushButton('Start')
        stop_button = QPushButton('Stop')

        start_button.clicked.connect(self.vid.startVideo)
        start_button.setFixedWidth(100)
        start_button.setFixedHeight(90)
        stop_button.clicked.connect(self.disable)
        stop_button.setFixedWidth(100)
        stop_button.setFixedHeight(90)

        # void QGridLayout::addWidget(QWidget * widget, int fromRow, int fromColumn, int rowSpan, int columnSpan, Qt::Alignment alignment = 0)

        # layout.setContentsMargins(left, top, right, bottom)
        layout.setContentsMargins(75, 100, 75, 50)
        layout.addWidget(image_viewer, 1, 1, 2, 2)
        # layout.addStretch()
        layout.addWidget(start_button, 3, 0, 2, 1)
        layout.addWidget(stop_button, 3, 1, 2, 1)


        # # button to record
        # record_btn = QPushButton('Record', self)
        # record_btn.setToolTip('Press to record with your camera')
        # record_btn.clicked.connect(self.Record)
        # record_btn.resize(record_btn.sizeHint())
        # layout.addWidget(record_btn)


        # 3d reconstruction
        reconstruct_btn = QPushButton('Reconstruct', self)
        reconstruct_btn.clicked.connect(self.Reconstruct)
        # reconstruct_btn.resize(reconstruct_btn.sizeHint())
        reconstruct_btn.setFixedWidth(100)
        reconstruct_btn.setFixedHeight(90)
        layout.addWidget(reconstruct_btn, 3, 2, 2, 1)

        # Augmentation
        augment_btn = QPushButton('Augment', self)
        augment_btn.clicked.connect(self.Augment)
        augment_btn.resize(reconstruct_btn.sizeHint())
        augment_btn.setFixedWidth(100)
        augment_btn.setFixedHeight(90)
        layout.addWidget(augment_btn, 3, 3, 2, 1)

        # set geometry
        self.setGeometry(0, 0, 1000, 600)
        self.setWindowTitle('App')
        self.show()



    def disable(self):
        print ('sending stop signal to the worker object')
        self.sgnStop.emit()  # send a queuedconnection type signal to the worker, because its in another thread

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
                self.realsense_recorder(path_folder)
            else:
                pass


    def Reconstruct(self):
        self.r = Reconstructor()

    def Segment(self):
        self.s = Segmenter(self.r.reconstructed_pointcloud)

    def Augment(self):
        name = "/home/pti/Downloads/tum/at3dcv/project/pointclouds_for_fun/scene0000_00_vh_clean_2.labels.ply"
        dummy_pcl = read_point_cloud(name)
        draw_geometries([dummy_pcl])
        # self.t = Augmentor(self, dummy_pcl)




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
        print("ShowVideo")

    @pyqtSlot()
    def startVideo(self):

        print("start videoooooooooooooooooo")
        self._running = True
        config = "Reconstruction/ReconstructionSystem/config/realsense.json"
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

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

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
                # print(height, width)

                qt_image = QImage(color_swapped_image,
                                        width,
                                        height,
                                        color_swapped_image.strides[0],
                                        QImage.Format_RGB888).scaledToWidth(854)


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
    view = View()
    sys.exit(app.exec_())

