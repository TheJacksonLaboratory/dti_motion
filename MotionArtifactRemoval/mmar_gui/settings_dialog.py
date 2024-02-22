# Copyright The Jackson Laboratory, 2022
# authors: Jim Peterson, Abed Ghanbari
'''
Implimentation of the SettingsDialog class
'''
import os
import pickle
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFrame, QGroupBox, QMessageBox, QLabel, QCheckBox, QDoubleSpinBox,
    QComboBox)
from .project_file import MmarProjectFile
from .utils import show_warning


class SettingsDialog(QDialog):
    '''
    This class, subclassed from QDialog, implements the Settings dialog.
    '''

    def __init__(self, def_model_file, initial_settings=None, parent=None):
        '''
        Initialize the modal Settings dialog.
        '''
        super().__init__(parent)
        self.setMinimumWidth(500)

        self.settings_dict = {}
        if not initial_settings is None:
            self.settings_dict = initial_settings

        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowTitle("mmar_gui - settings")

        # Create the main layout
        main_layout = QVBoxLayout()

        # Create the form layout (add to main layout)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # input directory (add to form)
        self.e_input_dir = QLineEdit()
        self.e_input_dir.setText(self.settings_dict['input_dir'])
        self.e_input_dir.setReadOnly(True)
        self.e_input_dir.setToolTip('Top directory of project.')
        b_input_dir = QPushButton("...")
        b_input_dir.setToolTip("Click to select input directory.")
        b_input_dir.clicked.connect(self.on_select_input_dir)
        hbl = QHBoxLayout()
        hbl.addWidget(self.e_input_dir, 1)
        hbl.addWidget(b_input_dir, 0)
        form.addRow("input directory", hbl)

        # image filename (add to form layout)
        self.e_image_filename = QLineEdit()
        self.e_image_filename.setText(self.settings_dict['image_filename'])
        self.e_image_filename.setToolTip("Name of image file.")
        form.addRow("image filename", self.e_image_filename)

        # Create button layout (add to main layout)
        button_box = QHBoxLayout()

        b_ok = QPushButton("ok")
        b_ok.setToolTip("Click to accept new settings.")
        b_ok.clicked.connect(self.on_ok)
        button_box.addWidget(b_ok, 0)

        b_cancel = QPushButton("cancel")
        b_cancel.setToolTip("Click to return without changing settings.")
        b_cancel.clicked.connect(self.on_cancel)
        button_box.addWidget(b_cancel, 0)

        b_save = QPushButton("save...")
        b_save.setToolTip("Click to save setting to a file.")
        b_save.clicked.connect(self.on_save)
        button_box.addWidget(b_save, 0)

        b_load = QPushButton("load...")
        b_load.setToolTip("Click to load settings from a file.")
        b_load.clicked.connect(self.on_load)
        button_box.addWidget(b_load, 0)

        # Create separator line between form and buttons
        line = QFrame()
        line.setFrameStyle(QFrame.HLine)

        # bval/bvec groupbox
        self.meta_groupbox = QGroupBox("bval/bvec metadata")
        tool_tip = "When selected the tensor metatdata files will be replicated and edited for each slice."
        tool_tip += "\nAdditionally, the bval file will be used to determine the longest scan to process."
        self.meta_groupbox.setToolTip(tool_tip)
        self.meta_groupbox.setCheckable(True)
        self.meta_groupbox.setChecked(self.settings_dict['use_metadata'])
        vbl = QVBoxLayout()
        self.meta_groupbox.setLayout(vbl)
        # bval filename (add to form layout)
        self.e_bval_filename = QLineEdit()
        self.e_bval_filename.setText(self.settings_dict['bval_filename'])
        self.e_bval_filename.setToolTip("Name of bval file.")
        vbl.addWidget(self.e_bval_filename)
        # bvec filename (add to form layout)
        self.e_bvec_filename = QLineEdit()
        self.e_bvec_filename.setText(self.settings_dict['bvec_filename'])
        self.e_bvec_filename.setToolTip("Name of bvec file")
        vbl.addWidget(self.e_bvec_filename)

        output_groupbox = QGroupBox("output")
        vbl = QVBoxLayout()
        output_groupbox.setLayout(vbl)
        # output directory (add to form)
        self.e_output_dir = QLineEdit()
        self.e_output_dir.setText(self.settings_dict['output_dir'])
        self.e_output_dir.setToolTip(
            'Name of directory for output files.\nThe directory will be created in each mouse directory.')
        hbl = QHBoxLayout()
        hbl.addWidget(QLabel('directory'), 0)
        hbl.addWidget(self.e_output_dir, 1)
        vbl.addLayout(hbl, 1)
        self.c_use_subdirs = QCheckBox("use subdirectory per slice")
        self.c_use_subdirs.setChecked(self.settings_dict['use_subdirs'])
        tip = "When selected a subdirectory/folder is created for each slice.  In this case the output filenames will match the input.\n"
        tip += "When not selected the output directory is flat, with the slice indices prepended to the image filenames."
        self.c_use_subdirs.setToolTip(tip)
        vbl.addWidget(self.c_use_subdirs)
        self.c_save_4d = QCheckBox("save slices as 4D images")
        self.c_save_4d.setChecked(self.settings_dict['save_4d'])
        tip = "When selected the image file for each output slice will be saved as a 4D dataset, including all other slices.\n"
        tip += "For a given slice, the rejected frames for it will be removed for all other slices included in its image file.\n"
        tip += "This option is offered to accomodate down-stream tools that may require 4D images as input."
        self.c_save_4d.setToolTip(tip)
        vbl.addWidget(self.c_save_4d)

        model_groupbox = QGroupBox("model")
        vbl = QVBoxLayout()
        model_groupbox.setLayout(vbl)
        self.e_model_file = QLineEdit()
        self.e_model_file.setText(self.settings_dict['model_file'])
        self.e_model_file.setReadOnly(True)
        self.e_model_file.setToolTip("Alternative model (.pkl) file to use.")
        self.c_model = QCheckBox("alternate model (.plk):")
        tip = "When unchecked the default classification model is used.\n"
        tip += "When checked the specified trained model is used."
        self.c_model.setToolTip(tip)
        b_model_file = QPushButton("...")
        b_model_file.setToolTip("Click to select an alternate model file.")
        b_model_file.clicked.connect(self.on_select_model_file)
        hbl = QHBoxLayout()
        hbl.addWidget(self.c_model, 0)
        hbl.addWidget(self.e_model_file, 1)
        hbl.addWidget(b_model_file, 0)
        vbl.addLayout(hbl, 1)
        hbl = QHBoxLayout()
        self.choose_clf = QComboBox()
        hbl.addWidget(QLabel("classifier method"))
        hbl.addWidget(self.choose_clf)
        hbl.addStretch()
        vbl.addLayout(hbl)
        self.s_thresh = QDoubleSpinBox()
        self.s_thresh.setDecimals(3)
        self.s_thresh.setRange(0.0, 1.0)
        self.s_thresh.setSingleStep(0.05)
        self.s_thresh.setValue(self.settings_dict['ml_prob'])
        tip = "When frames are classified as certainty of classification for each frame, in the form of a probability"
        self.s_thresh.setToolTip(tip)
        hbl = QHBoxLayout()
        hbl.addWidget(QLabel("probability threshold:"))
        self.s_thresh.setMinimumWidth(75)
        hbl.addWidget(self.s_thresh, 0)
        hbl.addStretch()
        vbl.addLayout(hbl, 1)

        main_layout.addLayout(form, 1)
        main_layout.addWidget(self.meta_groupbox, 1)
        main_layout.addWidget(output_groupbox, 1)
        main_layout.addWidget(model_groupbox, 1)
        main_layout.addWidget(line)
        main_layout.addLayout(button_box, 0)
        self.setLayout(main_layout)

        # load the model file to determine the method choices
        use_alt = self.settings_dict.get('use_alt_model')
        if use_alt is None:
            use_alt = False
        model_file = self.settings_dict.get('model_file')
        if model_file is None:
            model_file = ""
        if not use_alt or not model_file:
            model_file = def_model_file

        model = pickle.load(open(model_file, 'rb'))
        method_list = list(model.keys())
        for method_str in method_list:
            self.choose_clf.addItem(method_str)

        b_ok.setFocus()


    def on_ok(self):
        ''' Called when the OK button is pressed. '''
        if (self.e_bval_filename.text() == '' or self.e_bvec_filename.text() == '') and self.meta_groupbox.isChecked():
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText("bval and bvec filenames must not be empty.")
            msg_box.setWindowTitle("mmar_gui")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
            return
        self.settings_dict['input_dir'] = self.e_input_dir.text()
        self.settings_dict['output_dir'] = self.e_output_dir.text()
        self.settings_dict['use_alt_model'] = self.c_model.isChecked()
        self.settings_dict['model_file'] = self.e_model_file.text()
        self.settings_dict['ml_prob'] = self.s_thresh.value()
        self.settings_dict['ml_classifier'] = self.choose_clf.currentText()
        self.settings_dict['image_filename'] = self.e_image_filename.text()
        self.settings_dict['bval_filename'] = self.e_bval_filename.text()
        self.settings_dict['bvec_filename'] = self.e_bvec_filename.text()
        self.settings_dict['use_metadata'] = self.meta_groupbox.isChecked()
        self.settings_dict['use_subdirs'] = self.c_use_subdirs.isChecked()
        self.settings_dict['save_4d'] = self.c_save_4d.isChecked()

        self.done(0)


    def on_cancel(self):
        ''' Called when the Cancel button is pressed. '''
        self.done(-1)

    def on_select_input_dir(self):
        ''' Called when the b_input_dir button is pressed. '''
        self.settings_dict['input_dir'] = QFileDialog.getExistingDirectory(self, "default", "./")
        self.e_input_dir.setText(self.settings_dict['input_dir'])

    def on_select_output_dir(self):
        ''' Called when the b_output_dir button is pressed. '''
        self.settings_dict['output_dir'] = QFileDialog.getExistingDirectory(self, "default", "./")
        self.e_output_dir.setText(self.settings_dict['output_dir'])

    def on_select_model_file(self):
        ''' Called when the b_model_file button is pressed. '''
        file_tup = QFileDialog.getOpenFileName(self, "Open file", "./", "MRI scans (*.pkl)")
        self.settings_dict['model_file'] = file_tup[0]
        self.e_model_file.setText(self.settings_dict['model_file'])


    def on_save(self):
        ''' Called when the button to save the settings is clicked '''
        filename_tup = QFileDialog.getSaveFileName(self, "Save Config File", "./mmar_config.txt")
        filename = filename_tup[0]
        if not filename:
            return

        settings_tmp = {}
        settings_tmp['input_dir'] = self.e_input_dir.text()
        settings_tmp['output_dir'] = self.e_output_dir.text()
        settings_tmp['use_alt_model'] = self.c_model.isChecked()
        settings_tmp['model_file'] = self.e_model_file.text()
        settings_tmp['ml_prob'] = self.s_thresh.value()
        settings_tmp['ml_classifier'] = self.choose_clf.currentText()
        settings_tmp['image_filename'] = self.e_image_filename.text()
        settings_tmp['use_metadata'] = self.meta_groupbox.isChecked()
        settings_tmp['bval_filename'] = self.e_bval_filename.text()
        settings_tmp['bvec_filename'] = self.e_bvec_filename.text()
        settings_tmp['use_subdirs'] = self.c_use_subdirs.isChecked()
        settings_tmp['save_4d'] = self.c_save_4d.isChecked()

        MmarProjectFile().save(settings_tmp, filename)



    def load_config(self, filename):
        ''' Loads the specified settings file '''
        if not filename:
            return
        if not os.path.exists(filename):
            return

        settings = MmarProjectFile().load(filename)

        self.e_input_dir.setText(settings['input_dir'])
        self.e_output_dir.setText(settings['output_dir'])
        self.c_model.setChecked(settings.get('use_alt_model') == 'True')
        self.e_model_file.setText(settings['model_file'])
        self.s_thresh.setValue(settings['ml_prob'])
        idx = self.choose_clf.findText(settings['ml_classifier'])
        if idx < 0:
            show_warning(msg="The project file does not specify a valid classifier method.")
        self.choose_clf.setCurrentIndex(self.choose_clf.findText(settings['ml_classifier']))
        self.e_image_filename.setText(settings['image_filename'])
        self.meta_groupbox.setChecked(settings['use_metadata'])
        self.e_bval_filename.setText(settings['bval_filename'])
        self.e_bvec_filename.setText(settings['bvec_filename'])
        self.c_use_subdirs.setChecked(settings['use_subdirs'])
        self.c_save_4d.setChecked(settings['save_4d'])


    def on_load(self):
        ''' Called when the button to load a settings file is clicked. '''
        filename_tup = QFileDialog.getOpenFileName(self, "Open Config File", "./mmar_config.txt")
        filename = filename_tup[0]
        if not filename:
            return
        self.load_config(filename)
