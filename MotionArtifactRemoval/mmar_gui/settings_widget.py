# Copyright The Jackson Laboratory, 2022
# authors: Jim Peterson, Abed Ghanbari
'''
Implimentation of the SettingsWidget class
'''
import os

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QWidget, QLineEdit, QPushButton, QHBoxLayout)
from .settings_dialog import SettingsDialog
from .project_file import MmarProjectFile


class SettingsWidget(QWidget):
    """
    Widget that holds the settings and controls the modification of the settings.
    """

    directoryOpen = QtCore.pyqtSignal(bool)

    def __init__(self, model_file):
        super().__init__()

        self.settings_dict = {}
        self.default_model_file = model_file

        self.subfolders = None

        self.e_input_dir = QLineEdit(self)
        self.e_input_dir.setReadOnly(True)

        self.prntfldr_button = QPushButton('Settings ...')
        self.prntfldr_button.clicked.connect(self.on_settings_dialog_clicked)

        layout = QHBoxLayout()
        layout.addWidget(self.e_input_dir)
        layout.addWidget(self.prntfldr_button)

        self.setLayout(layout)

        self.settings_dict = MmarProjectFile().load('mmar_project.txt')
        self.e_input_dir.setText(self.settings_dict['input_dir'])

        self.populate_project_directories()


    def populate_project_directories(self):
        '''
        Finds all mouse directories for the input directory.
        '''
        project_folder = self.settings_dict['input_dir']
        image_filename = self.settings_dict['image_filename']
        self.e_input_dir.setText(project_folder)

        self.subfolders = []
        if os.path.isdir(project_folder):
            self.subfolders = [f for f in os.listdir(project_folder) if not f.startswith('.')]
            self.subfolders = [f for f in self.subfolders if os.path.isdir(os.path.join(project_folder, f))]
            self.subfolders = [f for f in self.subfolders if os.path.isfile(os.path.join(project_folder, f, image_filename))]
        if len(self.subfolders)>0:
            self.directoryOpen.emit(True)
        else:
            self.directoryOpen.emit(False)


    def on_settings_dialog_clicked(self):
        """
        open folder
        """
        dlg = SettingsDialog(self.default_model_file, self.settings_dict)
        if dlg.exec() < 0:
            return

        self.populate_project_directories()
        MmarProjectFile().save(self.settings_dict, 'mmar_project.txt')


    def get_tensor_metadata_files(self):
        ''' return the bval/bvec filenames '''
        return self.settings_dict['bval_filename'], self.settings_dict['bvec_filename']

    def get_filename(self, mouse_name):
        '''
        Concatinate the project directory, mouse directory, and the image filename,
        creating a full path to the image file.

        Parameters:
            mouse_name - string
                name of the mouse directory (e.g. 'CK_DIF_0001_MUS00028472_1')
        Return:
            full path of image filename

        '''
        fname = os.path.join(self.settings_dict['input_dir'], mouse_name, self.settings_dict['image_filename'])
        return fname

    def get_project_folder(self):
        '''
        Returns the project folder where the input scans are found.
        '''
        return self.settings_dict['input_dir']

    def get_output_dir(self):
        '''
        Returns the output directory where the edited scans will be written.
        '''
        return self.settings_dict['output_dir']

    def get_list_of_mice(self):
        """
        get list of mouse names
        """
        return self.subfolders

    def get_use_metadata(self):
        ''' Return True if 'use_metadata" is checked. '''
        return self.settings_dict['use_metadata']

    def get_use_subdirs(self):
        ''' Return True if 'use_subdirs" is checked. '''
        return self.settings_dict['use_subdirs'] is True

    def get_save_4d(self):
        ''' Return True if 'save_4d" is checked. '''
        return self.settings_dict['save_4d'] is True

    def get_ml_thresh(self):
        '''
        Returns the selected probability threshold used for classifying rejected frames.
        '''
        return self.settings_dict['ml_prob']

    def get_classifier(self):
        '''
        Returns the selected classifier name as a string.  This must match a classifier represented in the model file.
        '''
        classifier_str = self.settings_dict.get('ml_classifier')
        if classifier_str is None:
            classifier_str = ''
        return classifier_str
