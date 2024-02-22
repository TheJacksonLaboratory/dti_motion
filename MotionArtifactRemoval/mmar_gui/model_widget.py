# Copyright The Jackson Laboratory, 2021
# authors: Jim Peterson, Abed Ghanbari
from PyQt5 import QtCore
from PyQt5.QtWidgets import QComboBox, QGridLayout, QLabel, QLineEdit, QWidget

from ..mmar.bad_frames import find_bad_frames_ml
from ..motion_detector.utils import get_rejected_frames_mouse_set


class ModelWidget(QWidget):

    modelSelectionChanged = QtCore.pyqtSignal(str)
    thresholdChanged = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()

        self.model_lbl = QLabel(self)
        self.model_lbl.setText('Model:')

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(100)

        self.update_combo()
        self.model_combo.currentTextChanged.connect(self.modelSelectionChanged)

        # layout
        layout = QGridLayout()
        layout.addWidget(self.model_lbl,0,0,1,1)
        layout.addWidget(self.model_combo,0,1,1,1)
        
        self.setLayout(layout)

    def update_combo(self):
        """
        update model selection combobox
        """
        self.model_combo.clear()
        self.model_combo.addItem('ML')
        self.model_combo.addItem('Expert')
        self.model_combo.setCurrentIndex(0)

    def get_rejected_frames(self,
                            parent_folder=None, 
                            mouse_name=None,
                            data=None, 
                            CV_result=None,
                            classifier="Logistic Regression",
                            threshold=0.5
                            ):
        """
        get rejected frames
        """
        Y_expert = get_rejected_frames_mouse_set(parent_folder, mouse_name)
        Y_predicted, Y_proba = find_bad_frames_ml(
            data,
            cv_result=CV_result,
            classifier=classifier,
            num_slices=data.shape[1],
            num_frames_large=data.shape[0]+1,
            prob_thresh=threshold
        )

        if self.model_combo.currentText() == 'ML':
            return Y_predicted, Y_proba
        
        if self.model_combo.currentText() == 'Expert':
            return Y_expert, Y_proba
        
