'''
Copyright The Jackson Laboratory, 2021
authors: Jim Peterson, Abed Ghanbari

This module implements the widget that displays the list of
accepted and rejected frames.
'''
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QWidget)

class FrameListWidget(QWidget):

    currentSelectionChanged = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_list1 = QLabel(self)
        self.label_list1.setText('Accepted')
        self.label_list2 = QLabel(self)
        self.label_list2.setText('Rejected')

        self.framelist_accepted = QListWidget()
        self.framelist_accepted.setMaximumWidth(150)
        self.framelist_accepted.setSortingEnabled(True)
        self.framelist_accepted.itemDoubleClicked.connect(
                self.framelist_accepted_double_clicked)
        self.framelist_accepted.itemSelectionChanged.connect(
                lambda: self.framelist_clicked(accepted_selected=True))

        self.framelist_rejected = QListWidget()
        self.framelist_rejected.setMaximumWidth(150)
        self.framelist_rejected.setSortingEnabled(True)
        self.framelist_rejected.itemDoubleClicked.connect(
                self.framelist_rejected_double_clicked)
        self.framelist_rejected.itemSelectionChanged.connect(
                lambda: self.framelist_clicked(accepted_selected=False))

        # button to save images
        self.b_save_all = QPushButton('Save All Slices')
        self.b_save_all.setToolTip("Save all slices with frames rejected by current settings removed.")
        self.b_save_all.clicked.connect(self.on_save_clicked)

        vbl_main = QVBoxLayout()
        vbl_1 = QVBoxLayout()
        vbl_2 = QVBoxLayout()
        hbl_lists = QHBoxLayout()
        vbl_1.addWidget(self.label_list1)
        vbl_1.addWidget(self.framelist_accepted)
        vbl_2.addWidget(self.label_list2)
        vbl_2.addWidget(self.framelist_rejected)
        hbl_lists.addLayout(vbl_1)
        hbl_lists.addLayout(vbl_2)
        vbl_main.addLayout(hbl_lists)
        vbl_main.addWidget(self.b_save_all)

        self.setLayout(vbl_main)


    def clear(self):
        '''
        Clears the accepted and rejected frame lists.
        '''
        self.framelist_accepted.clear()
        self.framelist_rejected.clear()


    def update_framelist(self, rejected_frames, bad_frames_probability, first_frame, frame_count):
        """
        update list of frames in accepted and rejected lists
        """
        self.framelist_accepted.clear()
        self.framelist_rejected.clear()
        for i in range(frame_count-1):
            prob = bad_frames_probability[i]
            item = f"{str(first_frame+i+1).zfill(2)} - {prob:.2f}"
            if i in rejected_frames:
                self.framelist_rejected.insertItem(i, item)
            else:
                self.framelist_accepted.insertItem(i, item)

    def framelist_clicked(self, accepted_selected=False):
        """
        Emits the currentSelectionChanged signal with the current selected
        """
        if accepted_selected:
            item = self.framelist_accepted.currentItem()
        else:
            item = self.framelist_rejected.currentItem()
        if item is not None:
            self.currentSelectionChanged.emit(int(item.text()[:2]))

    def framelist_accepted_double_clicked(self):
        """
        takes item from accepted list and puts it in rejected list
        """
        item = self.framelist_accepted.currentItem()
        self.framelist_rejected.addItem(item.text())
        self.framelist_accepted.takeItem(self.framelist_accepted.row(item))
        frame_number = int(item.text().split(' ')[0])
        self.parent().move_frame_to_rejected(frame_number)

    def framelist_rejected_double_clicked(self):
        """
        takes item from rejected list and puts it in accepted list
        """
        item = self.framelist_rejected.currentItem()
        self.framelist_accepted.addItem(item.text())
        self.framelist_rejected.takeItem(self.framelist_rejected.row(item))
        frame_number = int(item.text().split(' ')[0])
        self.parent().move_frame_from_rejected(frame_number)

    def get_accepted_frames(self):
        """
        returns a list of the accepted frames
        """
        return [int(self.framelist_accepted.item(i).text()[:2])
                        for i in range(self.framelist_accepted.count())]

    def get_rejected_frames(self):
        """
        returns a list of the rejected frames
        """
        return [int(self.framelist_rejected.item(i).text()[:2])
                        for i in range(self.framelist_rejected.count())]

    def on_save_clicked(self):
        '''
        Called when the "Save Modified Scan" button is clicked.
        '''
        self.parent().save_all_slices()

