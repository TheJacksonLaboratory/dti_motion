'''
Copyright The Jackson Laboratory, 2022
authors: Jim Peterson, Abed Ghanbari
'''
from PyQt5.QtWidgets import QMessageBox

def show_warning(msg='', window_title="MMAR Warning!"):
    '''
    shows a dialog box.
    '''
    msg_box = QMessageBox()
    msg_box.setText(msg)
    msg_box.setWindowTitle(window_title)
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec()
