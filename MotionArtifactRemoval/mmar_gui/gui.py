# Copyright The Jackson Laboratory, 2021
# authors: Jim Peterson, Abed Ghanbari
'''
This module implements the main application for the mmar GUI application.
QT is used for implementation of the user interface elements and for signal control.
To run after instalation of the module:
    python -m MotionArtifactRemoval.mmar_gui.gui

'''
import os
import pickle
import sys

from os.path import dirname as up
import SimpleITK as sitk
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QGridLayout, QWidget, QVBoxLayout, QMessageBox)

from ..mmar.bad_frames import create_mask
from ..mmar.version import Version
from ..mmar.tensor_metadata import TensorMetadata
from ..mmar.write_output import write_edited_data
from ..mmar.bad_frames import find_bad_frames_ml


from .canvas_widget import CanvasWidget
from .frame_list_widget import FrameListWidget
from .image_navigator_widget import ImageNavigator
from .image_info_widget import ImageInfoWidget
from .mouse_list_widget import MouseListWidget
from .settings_widget import SettingsWidget
from .utils import show_warning


class App(QWidget):
    ''' Main application for the mmar GUI. '''
    def __init__(self):
        super().__init__()
        self.title = f'MRI Motion Artifact Removal ({Version().str()})'

        # coordinates for the UI
        self.left = 10
        self.top = 10
        self.width = 600
        self.height = 1000
        self.frame_num = None
        self.data = None
        self.data_original = None
        self.file_name = ""
        self.tensor_metadata = TensorMetadata()

        self.probs_by_slice = [] # when populated the frame indices are 0-based, starting after the zero-diff frame.
        self.rejected_frames_by_slice = []
        self.mouse_name = ''
        self.parent_folder = ''

        self.first_frame = 0
        self.frame_count = 0

        # Load motion artifact frames
        self.update_model()

        self.init_ui()

    def move_frame_to_rejected(self, frame_number):
        '''
        Called to move a frame to the rejected list.
        The frame_number is the 0-based frame in the scane.
        '''
        frame_adjusted = frame_number - 1
        slice_number = self.image_navigator.get_current_slice()
        self.rejected_frames_by_slice[slice_number].append(frame_adjusted)
        self.rejected_frames_by_slice[slice_number].sort()
        return

    def move_frame_from_rejected(self, frame_number):
        '''
        Called to remove a frame from the rejected list.
        The frame_number is the 0-based frame in the scane.
        '''
        frame_adjusted = frame_number - 1
        slice_number = self.image_navigator.get_current_slice()
        try:
            self.rejected_frames_by_slice[slice_number].remove(frame_adjusted)
        except:
            return


    def init_ui(self):
        '''
        initialize the UI
        we use a grid structure for the app
        '''

        # window
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # widgets
        self.image_navigator = ImageNavigator()
        self.frame_list_widget = FrameListWidget()
        self.mouse_list_widget = MouseListWidget()
        self.image_info_widget = ImageInfoWidget()
        self.canvas = CanvasWidget()
        model_path = os.path.join(up(up(__file__)),'motion_detector','trained_models','frame_rejection_all_80.pkl')
        self.settings_widget = SettingsWidget(model_path)

        # signals
        self.image_navigator.frameChanged[int].connect(self.update_canvas)
        self.image_navigator.sliceChanged[int].connect(self.update_canvas)
        self.image_navigator.sliceChanged[int].connect(self.update_framelist)
        self.image_navigator.setContrastChecked.connect(self.update_canvas)
        self.image_navigator.contrastRangeChanged.connect(self.update_canvas)
        self.image_navigator.imageOrMaskChanged.connect(self.update_canvas)

        self.frame_list_widget.currentSelectionChanged.connect(
                self.image_navigator.set_frame_value)

        self.mouse_list_widget.mouseSelectionChanged.connect(self.update_mouse)
        self.mouse_list_widget.mouseSelectionChanged.connect(self.update_framelist)

        self.settings_widget.directoryOpen.connect(
            lambda: self.mouse_list_widget.update_list(
                self.settings_widget.get_list_of_mice()
                )
            )

        # layout
        _canvas_length = 35
        layout = QGridLayout()
        layout.addWidget(self.settings_widget, 0, 1, 1, 5)

        vbl = QVBoxLayout()
        vbl.addWidget(self.mouse_list_widget, 1)
        vbl.addWidget(self.image_info_widget, 0)
        layout.addLayout(vbl, 1, 0, _canvas_length+3, 1)

        layout.addWidget(self.canvas, 2, 1, _canvas_length-1, 5)
        layout.addWidget(self.frame_list_widget, 1, 6, _canvas_length+3, 1)
        layout.addWidget(self.image_navigator, _canvas_length+1, 1, 1, 5)
        self.setLayout(layout)
        self.show()
        self.mouse_list_widget.update_list(self.settings_widget.get_list_of_mice())

    def update_mouse(self):
        '''
        Called when the mouse selection changes in the mouse list.
        '''
        self.update_filename()
        mouse = self.mouse_list_widget.get_selected_mouse()
        if not mouse:
            self.image_info_widget.clear()
            self.frame_list_widget.clear()
            self.canvas.clear()
            return

        self.update_data()
        self.image_navigator.update_navigator(
            max_frame_value=self.data_original.shape[0],
            max_slice=self.data_original.shape[1]
        )
        self.update_canvas()

    def update_filename(self):
        '''
        Updates the parent_folder, mouse_name, and file_name from the settings.
        JGP - Is this really needed?
        '''
        self.parent_folder = self.settings_widget.get_project_folder()
        self.mouse_name = self.mouse_list_widget.get_selected_mouse()
        self.file_name = self.settings_widget.get_filename(self.mouse_name)


    def update_data(self):
        """
        update the data, change the contrast range, and rejected frames
        """
        selected_mouse = self.mouse_list_widget.get_selected_mouse()
        if (not selected_mouse) or (not self.file_name):
            return

        # Check for the source file
        if not os.path.isfile(self.file_name):
            msg = f"File doesn't exist:\n    {self.file_name}"
            print('warning: '+msg)
            show_warning(msg=msg)
            return

        # Read the image file
        self.data_original = sitk.GetArrayFromImage(sitk.ReadImage(self.file_name))

        image_shape = self.data_original.shape
        if len(image_shape) != 4:
            msg = f"Image doesn't have the expected 4 dimensions:\n    {self.file_name}\n"
            print('warning: '+msg)
            show_warning(msg=msg)
            return

        scan_count = 1
        self.first_frame = 0
        self.frame_count = image_shape[0]
        if self.settings_widget.get_use_metadata():
            bval_filename, bvec_filename = self.settings_widget.get_tensor_metadata_files()
            if not self.tensor_metadata.find_files(self.file_name, bval_filename, bvec_filename):
                msg =   "warning - tensor metadata files don't exist:\n"
                msg += f"    {bval_filename}\n"
                msg += f"    {bvec_filename}"
                show_warning(msg=msg)
            else:
                scan_count = self.tensor_metadata.scan_count()
                self.first_frame, self.frame_count = self.tensor_metadata.get_longest_scan()

        self.data = self.data_original[self.first_frame:self.first_frame+self.frame_count+1,:,:,:]
        self.image_info_widget.set_info(image_shape[3], image_shape[2], image_shape[1], image_shape[0], scan_count, self.first_frame, self.first_frame+self.frame_count-1)
        # set range limits
        self.image_navigator.set_contrast_range([0,self.data.max()])

        # update rejected frames
        if not self.update_frame_rejection_prob():
            return
        self.update_framelist()


    def update_canvas(self):
        """
        update the image viewer
        """
        if self.data_original is None:
            return
        # contrast range
        if self.image_navigator.set_contrast.isChecked():
            contrast_range = self.image_navigator.get_contrast_range()
        else:
            contrast_range = None

        if self.image_navigator.image_or_mask.isChecked():
            # display mask
            image = create_mask(
                                self.data_original[0,self.image_navigator.get_current_slice(),:,:],
                                mask_is_fg=False
                            )[0]
        else:
            # display image
            image = self.data_original[self.image_navigator.get_frame_value(),
                        self.image_navigator.get_current_slice(),:,:]

        self.canvas.update_image(
                                image,
                                contrast_range=contrast_range,
                                display_mask=self.image_navigator.image_or_mask.isChecked()
                            )

    def update_frame_rejection_prob(self):
        """
        update the probability of rejecting a frame
        """
        if self.data is None:
            return False

        classifier_str = self.settings_widget.get_classifier()
        if not classifier_str:
            show_warning(msg="A classifier method supported by the model file must be selected.\n\nSee 'Settings.../model/classifier method'.")
            return False

        thresh = self.settings_widget.get_ml_thresh()

        self.rejected_frames_by_slice, self.probs_by_slice = self.get_rejected_frames(
                    parent_folder=self.parent_folder,
                    mouse_name=self.mouse_name,
                    data = self.data[1:,...],
                    CV_result=self.model,
                    classifier=classifier_str,
                    threshold=thresh,
                    )
        return True


    def update_framelist(self):
        '''
        Display the updated list of frames to be displayed.
        '''
        if (self.data is None) or (not self.mouse_list_widget.get_selected_mouse()):
            self.frame_list_widget.clear()
            return

        # update frame list with probabilites for current slice
        if not self.rejected_frames_by_slice:
            return
        rejected_frames = self.rejected_frames_by_slice[self.image_navigator.get_current_slice()]
        frame_probabilities = self.probs_by_slice[self.image_navigator.get_current_slice()]

        self.frame_list_widget.update_framelist(
            rejected_frames,
            frame_probabilities,
            self.first_frame,
            self.frame_count
        )


    def update_model(self, model_path=None):
        '''
        Load the model used for predictions.
        '''
        if model_path is None:
            model_path = os.path.join(up(up(__file__)),'motion_detector','trained_models','frame_rejection_all_80.pkl')
        self.model = pickle.load(open(model_path, 'rb'))



    def save_all_slices(self):
        '''
        Save all modified slices of the current image to the output directory.
        '''

        output_dir = self.settings_widget.get_output_dir()
        if not output_dir:
            output_dir = 'output'
        src_img_path = self.file_name
        output_dir = os.path.join(
            self.settings_widget.get_project_folder(),
            self.mouse_list_widget.get_selected_mouse(),
            self.settings_widget.get_output_dir())
        rejected_frames = [[i+1 for i in s] for s in self.rejected_frames_by_slice]

        use_subdirs = self.settings_widget.get_use_subdirs()
        save_3d = not self.settings_widget.get_save_4d()

        error_str = write_edited_data(
            output_dir,
            src_img_path,
            self.data_original,
            self.tensor_metadata,
            rejected_frames,
            use_subdirs,
            save_3d)
        if error_str:
            show_warning(msg=f"Error occurred writing data:\n\n{error_str}")
            return False

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(f"Modified Images saved in:\n\n{output_dir}")
        msg_box.setWindowTitle("mmar_gui")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setSizeGripEnabled(True)
        msg_box.exec()

        return True


    def get_rejected_frames(self,
                            parent_folder=None,
                            mouse_name=None,
                            data=None,
                            CV_result=None,
                            classifier="Logistic Regression",
                            threshold=0.5
                            ):
        '''
        get rejected frames
        '''
        Y_predicted, Y_proba = find_bad_frames_ml(
            data,
            cv_result=CV_result,
            classifier=classifier,
            num_slices=data.shape[1],
            num_frames_large=data.shape[0]+1,
            prob_thresh=threshold
        )

        return Y_predicted, Y_proba



def run_gui():
    '''
    Main entry point for the application
    '''
    app = QApplication(sys.argv)
    path = os.path.join(up(sys.modules[__name__].__file__), 'icon.png')
    app.setWindowIcon(QIcon(path))
    App()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_gui()
