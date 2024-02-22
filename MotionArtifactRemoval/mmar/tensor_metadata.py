# Copyright The Jackson Laboratory, 2022
# authors: Jim Peterson, Abed Ghanbari
'''
Implimentation of the calss TensorMetadata
'''
import os
from ..mmar.message_log import MessageLog

class TensorMetadata:
    '''
    This class implements methods for reading and writing dti metadata files.
    '''
    def __init__(self, log=MessageLog(print)):
        self.bval_file = ""
        self.bval_path = ""
        self.bval_list = []
        self.bval_separator = '\t'
        self.bvec_file = ""
        self.bvec_path = ""
        self.bvec_list = []
        self.bvec_separator = '\t'
        self.scan_list = []
        self.ignore_meta = False
        self.log = log

    def use_bval_bvec_files(self, bval_path, bvec_path):
        if not bval_path or not bvec_path:
            return False

        path_split = os.path.split(bval_path)
        self.bval_path = bval_path
        self.bval_file = path_split[1]

        path_split = os.path.split(bvec_path)
        self.bvec_path = bvec_path
        self.bvec_file = path_split[1]

        self.read_data()
        return True


    def find_files_with_default_names(self, scan_path):
        self.find_bval_file(scan_path)
        self.find_bvec_file(scan_path)
        if not self.bval_path or not self.bvec_path:
            return False
        self.read_data()
        return True

    def find_files(self, scan_path, bval_pattern, bvec_pattern):
        path_split = os.path.split(scan_path)
        if path_split[0]:
            input_dir = path_split[0]
        else:
            input_dir = "."

        bval_path = os.path.join(input_dir, bval_pattern)
        bvec_path = os.path.join(input_dir, bvec_pattern)

        if os.path.isfile(bval_path):
            self.bval_path = bval_path
            self.bval_file = bval_pattern
        else:
            self.log.add_msg(f"bval file does not exist: {bval_path}")
            self.bval_path = ""
            self.bval_file = ""
            return False

        if os.path.isfile(bvec_path):
            self.bvec_path = bvec_path
            self.bvec_file = bvec_pattern
        else:
            self.log.add_msg(f"bvec file does not exist: {bvec_path}")
            self.bvec_path = ""
            self.bvec_file = ""
            return False

        self.read_data()
        return True


    def ignore_metadata(self, expected_frames):
        self.ignore_meta = True
        self.scan_list = [(0, override_frames)]


    def read_data(self, bval_thresh=10):
        '''
        Read the bval and bvec files identified when the class was instantiated.
        Parameters:
            bval_thresh : int
                threshold used for determining zero-diffusion frames.
                This is an optional Parameters.
        Return:
            True if successful, False otherwise
        '''
        ret1 = self.read_bval_file()
        ret2 = self.read_bvec_file()
        if not (ret1 and ret2):
            return False

        if len(self.bval_list) != len(self.bvec_list):
            self.log.add_msg("bval and bvec files do not match length.")
            return False

        self.identify_scans(bval_thresh)

        return True


    def scan_count(self):
        '''
        Returns the number of scans in the nii file.
        Each scan is characterized by a start with one or more zero-diffusion frames.
        '''
        return len([p for p in self.scan_list if p[1] > 1])


    def find_bval_file(self, scan_path):
        '''
        Finds bval file in the original scan directory.
        If the bval file is not specified it is discovered in the input
        directory.  The following files are checked for existence:
            <input_dir>/<base_file_name>.bval
            <input_dir>/<base_file_name>.bvals
            <input_dir>/data.bval
            <input_dir>/data.bvals
        where <base_file_name> is the base name of the image file, or the
        base file name with "_0" removed, if it is there.
        Params:
            scan_path: string
                path name of the image file
        Return:
            True if successful
        '''
        path_split = os.path.split(scan_path)
        if path_split[0]:
            input_dir = path_split[0]
        else:
            input_dir = "."
        scan_file = path_split[1]

        i = scan_file.rfind('.nii')
        if i <= 0:
            i = scan_file.rfind('.tif')
        if i <= 0:
            return False
        base_input_file = scan_file[:i]

        possible_bval_names = []
        possible_bval_names.append(base_input_file+".bval")
        possible_bval_names.append(base_input_file+".bvals")
        possible_bval_names.append("data"+".bval")
        possible_bval_names.append("data"+".bvals")
        if base_input_file.endswith("_0"):
            possible_bval_names.append(base_input_file[:-2]+".bval")
            possible_bval_names.append(base_input_file[:-2]+".bvals")

        for file in possible_bval_names:
            path = os.path.join(input_dir, file)
            if os.path.isfile(path):
                self.bval_path = path
                self.bval_file = file
                break

        if self.bval_path  == "":
            return False

        return True


    def find_bvec_file(self, nii_file_path):
        '''
        Finds bvec file in the original scan directory.
        If the bvec file is not specified it is discovered in the input
        directory.  The following files are checked for existence:
            <input_dir>/<base_file_name>.bvec
            <input_dir>/<base_file_name>.bvecs
            <input_dir>/data.bvec
            <input_dir>/data.bvecs
        where <base_file_name> is the base name of the image file, or the
        base file name with "_0" removed, if it is there.
        Params:
            scan_path: string
                path name of the image file
        Return:
            True if successful
        '''

        path_split = os.path.split(nii_file_path)
        if path_split[0]:
            base_dir = path_split[0]
        else:
            base_dir = "."
        scan_file = path_split[1]

        i = scan_file.rfind('.nii')
        if i <= 0:
            i = scan_file.rfind('.tif')
        if i <= 0:
            return False
        base_input_file = scan_file[:i]

        possible_bvec_names = []
        possible_bvec_names.append(base_input_file+".bvec")
        possible_bvec_names.append(base_input_file+".bvecs")
        possible_bvec_names.append("data"+".bvec")
        possible_bvec_names.append("data"+".bvecs")
        if base_input_file.endswith("_0"):
            possible_bvec_names.append(base_input_file[:-2]+".bvec")
            possible_bvec_names.append(base_input_file[:-2]+".bvecs")

        for file in possible_bvec_names:
            path = os.path.join(base_dir, file)
            if os.path.isfile(path):
                self.bvec_path = path
                self.bvec_file = file
                break

        if self.bvec_path == "":
            return False

        return True


    def read_bval_file(self):
        '''
        Opens the bval file and populates the class variables:
            bval_list:
                A list floats, one per frame
            bvec_separator:
                Charater used to separate fields.  Must be tab, comma or space.

        Parameters:
            none
        Returns:
            True if successful, False otherwise
        '''
        with open(self.bval_path, encoding="latin-1") as file:
            lines = file.readlines()
        if len(lines) < 1:
            self.log.add_msg(
            f"error: input meta file is either empty or can not be opened ({self.bval_path}")
            return False
        if len(lines) > 1:
            self.log.add_msg(f"warning: Too many lines in metadata file. ({self.bval_path})")
        line = lines[0]
        if line[-1] == '\n':
            line = line[:-1]
        if '\t' in line:
            self.bval_separator = '\t'
        elif ',' in line:
            self.bval_separator = ','
        else:
            self.bval_separator = ' '
        str_list = line.split(self.bval_separator)
        self.bval_list = [float(s) for s in str_list]

        return True


    def read_bvec_file(self):
        '''
        Opens the bvec file and populates the class variables:
            bvec_list:
                A list of tuples containing the x, y, and z components of the bvecs
            bvec_separator:
                Charater used to separate fields.  Must be tab, comma or space.

        Parameters:
            none
        Returns:
            True if successful, False otherwise
        '''
        with open(self.bvec_path, encoding="latin-1") as file:
            lines = file.readlines()
        if len(lines) < 3:
            self.log.add_msg(
            f"error: too few lines in metadata file. ({self.bvec_path})")
            return False
        if len(lines) > 3:
            self.log.add_msg(f"warning: too many input lines in metadata file. ({self.bvec_path})")

        if lines[0][-1] == '\n':
            lines[0] = lines[0][:-1]
        if lines[1][-1] == '\n':
            lines[1] = lines[1][:-1]
        if lines[2][-1] == '\n':
            lines[2] = lines[2][:-1]
        line = lines[0]
        if '\t' in line:
            self.bvec_separator = '\t'
        elif ',' in line:
            self.bvec_separator = ','
        else:
            self.bvec_separator = ' '

        x_str = lines[0].split(self.bvec_separator)
        y_str = lines[1].split(self.bvec_separator)
        z_str = lines[2].split(self.bvec_separator)

        x_floats = [float(s) for s in x_str if s != '']
        y_floats = [float(s) for s in y_str if s != '']
        z_floats = [float(s) for s in z_str if s != '']

        if (len(x_floats) != len(y_floats)) \
        or (len(x_floats) != len(z_floats)) \
        or (len(x_floats) == 0):
            self.log.add_msg(f"error: number of x, y and z components does not agree. ({self.bvec_path})")
            return False

        self.bvec_list = [(x_floats[i], y_floats[i], z_floats[i]) for i in range(len(x_floats))]

        return True


    def identify_scans(self, bval_thresh):
        '''
        Identify the index of the start of scans, indicated by zero-diffusion
        frames in the bval file.
        Parameters:
            bval_thresh : float
                Threshold value, below which, indicates a zero-diffusion frames
                in the bval file.  The results are saved in the class variable
                scan_list.  The scan list is a list of tuples with the first
                coordinate being the start frame and the second coordinate, the length.
                The scan_list is sorted by desending length of scan.
            Return:
                none
        '''
        self.scan_list = []
        for idx in range(len(self.bval_list)):
            if (self.bval_list[idx] > bval_thresh) and (idx == 0 or self.bval_list[idx-1] <= bval_thresh):
                frame_0 = idx - 1
                if frame_0 < 0:
                    frame_0 = 0
                self.scan_list.append([frame_0, -1])
                continue
            if (self.bval_list[idx] > bval_thresh) and ((idx == len(self.bval_list)-1) or (self.bval_list[idx+1] <= bval_thresh)):
                self.scan_list[-1][1] = idx - self.scan_list[-1][0] + 1
        self.scan_list.sort(reverse=True, key = lambda x: x[1])


    def get_longest_scan(self):
        if len(self.scan_list) > 0:
            return self.scan_list[0]
        else:
            return [0, 0]


    def write(self, output_dir, prefix, rejected_frames):
        '''
        Write the metadata files after removing rejected frames.
        Parameters :
            output_dir : string
                directory where files will be written
            prefix : string
                When the output directory is flat, this prefix is prepended
                to all file names.  However, when there is a subdirectory per
                slice, this prefix is expeced to be empty.
        Return : bool
            True when successful
        '''
        if self.ignore_meta or len(self.bval_list)==0 or len(self.bvec_list)==0:
            return True

        bval_path = os.path.join(output_dir, prefix+self.bval_file)
        bval_noddi_path = os.path.splitext(bval_path)[0] + '_noddi' + os.path.splitext(bval_path)[1]
        bvec_path = os.path.join(output_dir, prefix+self.bvec_file)

        # Write bval file
        values = self.bval_list.copy()
        if len(rejected_frames):
            for i in sorted(rejected_frames, reverse=True):
                del values[i]
        bval_str = ""
        for val in values:
            bval_str += str(val) + self.bval_separator
        bval_str = bval_str[:-1] + '\n'
        with open(bval_path, 'w', encoding="latin-1") as file:
            file.write(bval_str)

        # Write bval NODDI file
        bval_str = ""
        for val in values:
            bval_str += str(100 * int(round(val/100.0))) + self.bval_separator
        bval_str = bval_str[:-1] + '\n'
        with open(bval_noddi_path, 'w', encoding="latin-1") as file:
            file.write(bval_str)

        # Write bvec file
        values = self.bvec_list.copy()
        if len(rejected_frames):
            for i in sorted(rejected_frames, reverse=True):
                del values[i]
        x_str = ""
        y_str = ""
        z_str = ""
        for val in values:
            x_str += str(val[0]) + self.bvec_separator
            y_str += str(val[1]) + self.bvec_separator
            z_str += str(val[2]) + self.bvec_separator
        x_str = x_str[:-1] + '\n'
        y_str = y_str[:-1] + '\n'
        z_str = z_str[:-1] + '\n'
        with open(bvec_path, 'w', encoding="latin-1") as file:
            file.write(x_str + y_str + z_str)

        return True
