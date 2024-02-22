# Copyright The Jackson Laboratory, 2022
# authors: Jim Peterson, Abed Ghanbari
'''
Implimentation of the ImageInfoWidget class
'''
from PyQt5.QtWidgets import (QWidget, QLabel, QFormLayout, QGroupBox, QHBoxLayout)

class ImageInfoWidget(QWidget):
    """
    Widget that holds the settings and controls the modification of the settings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.l_width = QLabel('')
        self.l_height = QLabel('')
        self.l_slices = QLabel('')
        self.l_frames = QLabel('')
        self.l_scans = QLabel('')
        self.l_longest_scan = QLabel('')

        form_layout = QFormLayout()
        form_layout.addRow("width:", self.l_width)
        form_layout.addRow("height:", self.l_height)
        form_layout.addRow("slices:", self.l_slices)
        form_layout.addRow("frames:", self.l_frames)
        form_layout.addRow("scans:", self.l_scans)
        form_layout.addRow("longest scan:", self.l_longest_scan)

        group_box = QGroupBox("Image Info")
        group_box.setLayout(form_layout)
        hbl = QHBoxLayout()
        hbl.addWidget(group_box)
        self.setLayout(hbl)


    def clear(self):
        ''' Clear all values in the display '''
        self.l_width.setText('')
        self.l_height.setText('')
        self.l_slices.setText('')
        self.l_frames.setText('')
        self.l_scans.setText('')
        self.l_longest_scan.setText('')


    def set_info(self, width, height, slices, frames, scans, start, end):
        '''
        Set all the values.
        Parameters:
            width : integer
                width of the image in pixels
            height : integer
                height of the image in pixels
            slices : integer
                number of slices in 3D scan
            frames : integer
                total number of frames in image file
            scans : integer
                number of scans found in image file, as determined by the bval metadata
            start : integer
                index of the first frame of the longest scan
            end : integer
                index of the last frame of the longest scan
        '''
        self.l_width.setText(f"{width}")
        self.l_height.setText(f"{height}")
        self.l_slices.setText(f"{slices}")
        self.l_frames.setText(f"{frames}")
        self.l_scans.setText(f"{scans}")
        self.l_longest_scan.setText(f"[{start}-{end}]")
