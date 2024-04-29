from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QGroupBox, QDoubleSpinBox

class ImageNavigator(QWidget):
    """
    widget to show frame
    """

    frameChanged = QtCore.pyqtSignal(int)
    sliceChanged = QtCore.pyqtSignal(int)
    contrastCheckBoxClicked = QtCore.pyqtSignal(bool)
    imageOrMaskChanged = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create layout and the three group boxes
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)
        frame_group = QGroupBox("slice/frame")
        self.contrast_group = QGroupBox("brightness/contrast")
        self.contrast_group.setCheckable(True)
        self.contrast_group.setChecked(False)
        self.contrast_group.clicked.connect(self.contrastCheckBoxClicked)
        vlayout.addWidget(frame_group)
        vlayout.addWidget(self.contrast_group)

        # slice/frame
        self.z_slice = QComboBox()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimumWidth(300)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged[int].connect(self.frameChanged)
        self.frame_slider.valueChanged[int].connect(self.set_label_frame_num)
        self.label_frame = QLabel("-")
        self.label_frame.setMinimumWidth(25)
        frame_layout = QHBoxLayout()
        frame_group.setLayout(frame_layout)
        frame_layout.addWidget(self.z_slice)
        frame_layout.addWidget(self.label_frame)
        frame_layout.addWidget(self.frame_slider)

        # contrast control
        self.contrast_spin_min = QDoubleSpinBox()
        self.contrast_spin_min.setDecimals(3)
        self.contrast_spin_min.setRange(0.0, 1.0)
        self.contrast_spin_min.setSingleStep(0.01)
        self.contrast_spin_min.setValue(0.0)
        self.contrast_spin_max = QDoubleSpinBox()
        self.contrast_spin_max.setDecimals(3)
        self.contrast_spin_max.setRange(0.0, 1.0)
        self.contrast_spin_max.setSingleStep(0.01)
        self.contrast_spin_max.setValue(1.0)
        contrast_layout = QHBoxLayout()
        self.contrast_group.setLayout(contrast_layout)
        contrast_layout.addWidget(QLabel('min:'))
        contrast_layout.addWidget(self.contrast_spin_min)
        contrast_layout.insertStretch(-1)
        contrast_layout.addWidget(QLabel('max:'))
        contrast_layout.addWidget(self.contrast_spin_max)
        contrast_layout.insertStretch(-1)
        self.contrast_spin_min.valueChanged.connect(self.contrast_control_changed)
        self.contrast_spin_max.valueChanged.connect(self.contrast_control_changed)
        return


    def set_frame_value(self, frame_num):
        self.frame_slider.setValue(frame_num)

    def get_frame_value(self):
        return self.frame_slider.value()
    
    def set_label_frame_num(self, value):
        self.label_frame.setText(str(value))

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
        min = self.contrast_spin_min.value()
        max = self.contrast_spin_max.value()
        return [min, max]

    def update_navigator(self, max_frame_value, max_slice):
        self.frame_slider.setMaximum(max_frame_value-1)
        self.frame_slider.setValue(1)
        self.label_z_update(max_slice)

    def get_current_slice(self):
        return self.z_slice.currentIndex()    

    def set_current_slice(self, slice_num):
        self.z_slice.setCurrentIndex(slice_num)
        self.z_slice.currentIndexChanged.connect(self.sliceChanged)

    def contrast_control_changed(self):
        '''
        This is called whenever either of the "brightness/contrast" controls are changed.
        '''
        self.parent().update_canvas()
        return

