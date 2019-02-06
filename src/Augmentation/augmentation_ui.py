import sys, glob, os
from PyQt5.QtWidgets import QApplication, QStackedWidget, QWidget, QListWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap

from collections import defaultdict

from open3d import *
from config import *

from paths import augmentation_util
sys.path.append(augmentation_util)
from utils import helper
from utils.icp_helper import get_registeration
from utils.vis_helper import *

 
class AugmentationUI(QtWidgets.QWidget):

    def __init__(self, augmentor):

        super().__init__()

        self.augmentor = augmentor
        self.furnitures = get_all_objects()
        self.source_object = ''
        self.source_object = ''

        self.leftlist = QListWidget()
        self.leftlist.insertItem(0, 'Show Point Cloud')
        self.leftlist.insertItem(1, 'Remove Object' )
        self.leftlist.insertItem(2, 'Show One Object' )
        self.leftlist.insertItem(3, 'Change Object' )
		
        self.stack1 = QtWidgets.QWidget()
        self.stack2 = QtWidgets.QWidget()
        self.stack3 = QtWidgets.QWidget()
        self.stack4 = QtWidgets.QWidget()

        self.ShowPointCloudWindow()
        self.RemoveObjectWindow()
        self.ShowOneObjectWindow()
        self.ChangeObjectWindow()
		
        self.Stack = QStackedWidget(self)
        self.Stack.addWidget(self.stack1)
        self.Stack.addWidget(self.stack2)
        self.Stack.addWidget(self.stack3)
        self.Stack.addWidget(self.stack4)

        label = QLabel(self)
        pixmap = QPixmap(logo_path)
        pixmap = pixmap.scaled(pixmap.width()/5, pixmap.height()/5)
        label.setPixmap(pixmap)
		
        hbox = QHBoxLayout(self)
        hbox.addWidget(label)
        hbox.addWidget(self.leftlist)
        hbox.addWidget(self.Stack)


        self.setLayout(hbox)
        self.leftlist.currentRowChanged.connect(self.display)
        self.setGeometry(150, 50, 1100, 400)
        self.setWindowTitle('AU.')

        self.show()


    def display(self,i):
        self.Stack.setCurrentIndex(i)

    def show_with_trajectory(self):
        custom_draw_geometry_with_camera_trajectory(self.augmentor.pointcloud)
    def show_labled_point_cloud(self):
        show_pcd([self.augmentor.labeld_pointcloud])
    def show_raw_point_cloud(self):
        show_pcd([self.augmentor.pointcloud])

    def show_one_object(self, object_name):
        color = objects_hash[object_name]
        t = self.augmentor.get_object_with_hashed_color(color)
        show_pcd([t])

    def remove_one_object(self, object_name):
        color = objects_hash[object_name]
        t = self.augmentor.remove_object_with_index(color)
        show_pcd([t])

    def ShowPointCloudWindow(self):

        RawButton = QtWidgets.QPushButton(self.stack1)
        RawButton.setText('Show Raw')
        RawButton.clicked.connect(self.show_raw_point_cloud)

        LabeldButton = QtWidgets.QPushButton(self.stack1)
        LabeldButton.setText('Show Labeld')
        LabeldButton.clicked.connect(self.show_labled_point_cloud)

        self.horizontalGroupBox = QGroupBox('', self.stack1)
        
        layout = QGridLayout()
        layout.setColumnStretch(5, 5)
        layout.addWidget(RawButton,0,1)
        layout.addWidget(LabeldButton, 2,1)

        self.horizontalGroupBox.setLayout(layout)

    def RemoveObjectWindow(self):

        objectChoice = QtWidgets.QLabel('Which Object You Want to Remove?', self.stack2)

        comboBox = QtWidgets.QComboBox(self.stack2)
        for item in self.augmentor.objects_dictionary_by_color:
            if(item in hashed_color_names.keys()):
                comboBox.addItem(hashed_color_names[item])

        comboBox.setCurrentIndex(0)
        comboBox.activated[str].connect(self.remove_one_object)


        self.horizontalGroupBox = QGroupBox('', self.stack2)
        layout = QGridLayout()
        
        layout.setColumnStretch(5, 5)
        layout.addWidget(objectChoice, 0, 0)
        layout.addWidget(comboBox, 0, 1)

        self.horizontalGroupBox.setLayout(layout)

    def ShowOneObjectWindow(self):
        
        objectChoice = QtWidgets.QLabel('Which Object You Want to See?', self.stack3)

        comboBox = QtWidgets.QComboBox(self.stack3)
        for item in self.augmentor.objects_dictionary_by_color:
            if(item in hashed_color_names.keys()):
                comboBox.addItem(hashed_color_names[item])

        comboBox.activated[str].connect(self.show_one_object)

        self.horizontalGroupBox = QGroupBox('', self.stack3)
        layout = QGridLayout()
        
        layout.setColumnStretch(5, 5)
        layout.addWidget(objectChoice, 0, 0)
        layout.addWidget(comboBox, 0, 1)

        self.horizontalGroupBox.setLayout(layout)

    def ChangeObjectWindow(self):

        objectChoice = QtWidgets.QLabel('Which Object You Want to Change?', self.stack4)

        comboBox = QtWidgets.QComboBox(self.stack4)
        for item in self.augmentor.objects_dictionary_by_color:
            if(item in hashed_color_names.keys()):
                comboBox.addItem(hashed_color_names[item])
        comboBox.setCurrentIndex(0)
        comboBox.activated[str].connect(self.__select_target_object)


        objectChoiceII = QtWidgets.QLabel('Which New Object You Want to Choose?', self.stack4)
        comboBoxII = QtWidgets.QComboBox(self.stack4)
        for key in self.furnitures.keys():
            i = 0
            for item in self.furnitures[key]:
                comboBoxII.addItem(key + ':' + str(i))
                i += 1
        comboBoxII.setCurrentIndex(0)
        comboBoxII.activated[str].connect(self.__select_source_object)


        ChangeButton = QtWidgets.QPushButton(self.stack4)
        ChangeButton.setText('Change')
        ChangeButton.clicked.connect(self.__change_object)
        
        self.horizontalGroupBox = QGroupBox('', self.stack4)
        
        layout = QGridLayout()
        layout.setColumnStretch(5, 5)
        layout.addWidget(objectChoice, 0, 0)
        layout.addWidget(comboBox, 0, 1)
        layout.addWidget(objectChoiceII, 2, 0)
        layout.addWidget(comboBoxII, 2, 1)
        
        layout.addWidget(ChangeButton, 2, 2)

        self.horizontalGroupBox.setLayout(layout)

    def __select_target_object(self, name):
        self.target_object = name
    def __select_source_object(self, name):
        self.source_object = name

    def __change_object(self):

        temp = self.source_object.split(':')
        source = self.furnitures[temp[0]][int(temp[1])]

        target = self.target_object

        [a, b] = self.augmentor.change_object(target, source)
        show_pcd([a, b])
    

def get_objects(name):
    os.chdir(furnitures_path + name)
    l = []
    for file in glob.glob("*.ply"):
        t = read_point_cloud(furnitures_path + name + file)
        l.append(t)
        if(len(l) > 3):
            break
            
    return l

def get_all_objects():
    t = defaultdict(list)
    t['table'] = get_objects('table/')
    t['chair'] = get_objects('chair/')
    t['bed'] = get_objects('bed/')

    return t