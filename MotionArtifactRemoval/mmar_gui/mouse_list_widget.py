'''
Copyright The Jackson Laboratory, 2022
authors: Jim Peterson, Abed Ghanbari

Implements the widget that displays the list of mice found in the project directory.
'''

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QLabel, QListWidget, QWidget

class MouseListWidget(QWidget):
    """
    widget to open folder
    """

    mouseListUpdated = QtCore.pyqtSignal(bool)
    mouseSelectionChanged = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mouse_list_lbl = QLabel(self) # label for list of mouse names
        self.mouse_list_lbl.setText('Mouse Names:')
        self.mouse_list_lbl.setAlignment(Qt.AlignBottom)

        # list of folders found
        self.mouselistbox = QListWidget()
        self.mouselistbox.setMinimumWidth(215)
        self.mouselistbox.setSortingEnabled(True)
        self.mouselistbox.itemSelectionChanged.connect(self.mouseSelectionChanged)

        # layout
        layout = QGridLayout()
        layout.addWidget(self.mouse_list_lbl,0,0,1,1)
        layout.addWidget(self.mouselistbox,1,0,40,1)
        self.setLayout(layout)

    def update_list(self, mouselist):
        """
        update list of mouse names
        """
        self.mouselistbox.clear()
        for mouse in mouselist:
            self.mouselistbox.addItem(mouse)

    def get_selected_mouse(self):
        """
        get selected mouse name
        """
        selection_list = self.mouselistbox.selectedItems()
        mouse_name = ''
        if len(selection_list) > 0:
            mouse_name = selection_list[0].text()
        return mouse_name
