from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QWidget, QGridLayout, QLabel, QComboBox, QCheckBox

from superqt import QLabeledRangeSlider

class ImageNavigator(QWidget):
    """
    widget to show frame
    """

    frameChanged = QtCore.pyqtSignal(int)
    sliceChanged = QtCore.pyqtSignal(int)
    setContrastChecked = QtCore.pyqtSignal(int)
    contrastRangeChanged = QtCore.pyqtSignal(tuple)
    imageOrMaskChanged = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # slider widget for the frames
        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.setFocusPolicy(Qt.NoFocus)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(10)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        
        self.frame_slider.valueChanged[int].connect(self.frameChanged)
        self.frame_slider.valueChanged[int].connect(lambda: self.set_label_frame_num(self.frame_slider.value()))
        
        self.label_z = QLabel(self)
        self.label_z.setText('Slice Number: ')

        self.z_slice = QComboBox()

        self.label_frame = QLabel(self)
        self.label_frame.setText('Frame Number')
        self.label_frame.setMinimumWidth(120)

        self.set_contrast = QCheckBox("Set Contrast",self)
        self.set_contrast.setMaximumWidth(120)
        self.set_contrast.stateChanged.connect(self.setContrastChecked)

        self.contrast_slider = QLabeledRangeSlider(Qt.Horizontal, self)
        self.contrast_slider.setEdgeLabelMode(0)
        self.contrast_slider.setFixedWidth(300)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.contrast_slider.setTickInterval(2**12)
        self.contrast_slider.setSingleStep(8)
        self.contrast_slider.setMinimum(0)
        self.contrast_slider.setMaximum(2**16)
        self.contrast_slider.setValue((0,2**14))
        self.contrast_slider.valueChanged.connect(self.contrastRangeChanged)

        self.image_or_mask = QCheckBox("Background Mask",self)
        self.image_or_mask.stateChanged.connect(self.imageOrMaskChanged)

        layout = QGridLayout()
        layout.addWidget(self.label_frame,0,0,1,1)
        layout.addWidget(self.frame_slider,0,1,1,5)
        layout.addWidget(self.label_z,0,6,1,1)
        layout.addWidget(self.z_slice,0,7,1,1)

        layout.addWidget(self.set_contrast,1,0,1,1)
        layout.addWidget(self.contrast_slider,1,1,1,5)
        layout.addWidget(self.image_or_mask,1,7,1,1)

        self.setLayout(layout)

    def set_frame_value(self, frame_num):
        self.frame_slider.setValue(frame_num)
    
    def get_frame_value(self):
        return self.frame_slider.value()
    
    def set_maximum(self, max_value):
        self.frame_slider.setMaximum(max_value-1)

    def set_label_frame_num(self, value):
        self.label_frame.setText('Frame Number: '+str(value))

    def label_z_update(self, nz):
        """
        update the z-slice label to include slice numbers
        """
        if self.z_slice.count()>0:
            self.z_slice.clear()
        for i in range(nz):
            self.z_slice.addItem(str(i))
        self.z_slice.setCurrentIndex(0)
        self.z_slice.currentIndexChanged.connect(self.sliceChanged)
 
    def get_contrast_range(self):
        return self.contrast_slider.sliderPosition()

    def set_contrast_range(self, contrast_range):
        self.contrast_slider.setMinimum(contrast_range[0])
        self.contrast_slider.setMaximum(contrast_range[1])

    def update_navigator(self, max_frame_value, max_slice):
        self.set_maximum(max_frame_value)
        self.label_z_update(max_slice)

    def get_current_slice(self):
        return self.z_slice.currentIndex()    

    def set_current_slice(self, slice_num):
        self.z_slice.setCurrentIndex(slice_num)
        self.z_slice.currentIndexChanged.connect(self.sliceChanged)

