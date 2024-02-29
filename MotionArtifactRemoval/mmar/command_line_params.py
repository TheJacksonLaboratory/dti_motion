# Copyright The Jackson Laboratory, 2022
# authors: Jim Peterson, Abed Ghanbari
'''
Implimentation of the class CommandLineParams
'''
import sys
import os
import argparse
import textwrap
import ast
from argparse import RawTextHelpFormatter
from datetime import datetime

from .version import Version


class CommandLineParams:
    ''' This class parses the command line parameters '''
    ClassifierMethods = (
        'MAD',
        'LogisticRegression',
        'SVM',
        'RandomForest',
        'AdaBoostedDecisionTree',
        'GradientBoostedDecisionTree',
        'GaussianNaiveBayes',
        )
    method_dict = {
        'MAD':'MAD',
        'LogisticRegression':'Logistic Regression',
        'SVM':'SVM',
        'RandomForest':'Random Forest',
        'AdaBoostedDecisionTree':'AdaBoosted decision tree',
        'GradientBoostedDecisionTree':'Gradient Boosted Decision Tree',
        'GaussianNaiveBayes':'Gaussian Naive Bayes',
        }

    scan_file = ""
    input_dir = "./"
    output_dir = "./results/"
    bval_file = ""
    bvec_file = ""
    scan_path = ""
    bval_thresh = 10.0
    rejected_frames_file = ""
    fg_thresh = 6.0
    bg_thresh = 3.5
    use_subdirs = False
    classifier_method = ClassifierMethods[3] # RandomForest
    ml_prob_thresh = 0.5
    ml_prob_thresh_exceptions = []
    save3d = False
    ignore_meta = False
    ml_model = ""


    def __init__(self):
        desc = "mmar - (MRI motion artifact removal) Identify and remove frames showing motion artifacts"
        epilog = textwrap.dedent(
        '''\
        classifier_method must be one of:
            MAD
            LogisticRegression
            SVM
            RandomForest
            AdaBoostedDecisionTree
            GradientBoostedDecisionTree
            GaussianNaiveBayes
            <file>
        -------------------------------------------------------------------------------------------------------
        The <input_scan> is a 4D image file (tif or nii.gz) containing a list of multi-frame slice images
        The "bvecs" and "bvals" files contain the corresponding diffusion tensor metadata.
        Both are text files with fields separated with either tabs or commas.
        The bvals file is a single record, while the bvecs file is three records for the x, y, z components
        separated.  Intensity values in the bvals file that are less than "bvals_threshold" are assumed to be
        non-diffusion frames, indicating the start of a new scan.

        When run successfully, the output directory will be populated with the following:
            <output_dir>
                log.txt
                excluded_frames.csv
                <slice directory>
                    <image file> - multi-frame slice image with rejected frames removed
                    <bval file> - copy of original bvals file with rejected frames removed
                    <noddi bval file> - same as <bval file> with values rounded to nearest 100
                    <bvec file> - copy of original bvecs file with rejected frames removed

        The file "excluded_frames.csv" contains comma-separated values, with one line per slice, listing
        the frames that were excluded from the original image.  In each record, the first column lists the
        slice index, followed by rejected frames.  NOTE: All indices are 0-based.

        -------------------------------------------------------------------------------------------------------
        '''
        )

        parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=RawTextHelpFormatter,
            epilog=epilog)
        parser.add_argument("input_scan", help="multi-frame 3D scan")
        parser.add_argument('-o', metavar='output_dir', default=self.output_dir,
            help="directory for results. Default="+self.output_dir)
        parser.add_argument('-bvec', metavar='bvec_file', default=self.bvec_file,
            help="input bvecs file - When not specified the scan file base + '.bvecs' is used")
        parser.add_argument('-bval', metavar='bval_file', default=self.bval_file,
            help="input bvals file - When not specified the scan file base + '.bvals' is used")
        parser.add_argument('-method', metavar='name', default=self.classifier_method,
            help="method for classifying frames. Default="+self.classifier_method)
        parser.add_argument('-n', metavar='bvals_threshold', type=float, default=self.bval_thresh,
            help="non-diffusion bval threshold, to determine beginning of scan (e.g. 10.0)")
        parser.add_argument('-fg', metavar='fg_thresh', type=float, default=self.fg_thresh,
            help="foreground threshold for MAD algorithm (e.g. 6.0)")
        parser.add_argument('-bg', metavar='bg_thresh', type=float, default=self.bg_thresh,
            help="background threshold for MAD algorithm (e.g. 3.5)")
        parser.add_argument('-subdirs', dest='subdirs', action='store_true',
            help="use subdirectory for each slice in output directory")
        parser.add_argument('-p', metavar='ml_prob', type=float, default=self.ml_prob_thresh,
            help="ML probability threshold [0.0 - 1.0].  Higher values reduce rejection rate.")
        parser.add_argument('-pe', metavar='"(frame, prob)"',
            help='comma separated list of ML probability exceptions by frame.  e.g. "(32, 0.75), (35, 0.80)"')
        parser.add_argument('-save3d', action='store_true',
            help="save the output slice images as multi-frame 2D images")
        parser.add_argument('-ignore_meta', action='store_true',
            help="ignore any bval and bvec metadata")
        parser.add_argument('-model', metavar='pkl-file', default=self.ml_model,
            help="model to use for the ML methods. A default model is used when none specified.")

        args = parser.parse_args()

        self.scan_path = args.input_scan
        self.output_dir = args.o
        self.bvec_file = args.bvec
        self.bval_file = args.bval
        self.bval_thresh = args.n
        self.fg_thresh = args.bg
        self.fg_thresh = args.fg
        self.classifier_method = args.method
        self.use_subdirs = args.subdirs
        self.ml_prob_thresh = args.p
        self.save3d = args.save3d
        self.ignore_meta = args.ignore_meta
        self.ml_model = args.model
        try:
            self.ml_prob_thresh_exceptions = ast.literal_eval('['+args.pe+']')
        except:
            if args.pe:
                print(f"error: malformed sensitivity exception list. ({args.pe})")

    def check_parameters(self):
        '''
        Check for required parameters and build full directory paths.
        If the metadata files are not specified, we look for the following files in order:
            <input_dir>/<base_file_name>.bvecs
            <input_dir>/<base_file_name>.bvec
            <input_dir>/data.bvecs
            <input_dir>/data.bvec
        and similarly
            <input_dir>/<base_file_name>.bvals
            <input_dir>/<base_file_name>.bval
            <input_dir>/data.bvals
            <input_dir>/data.bval
        '''
        path_split = os.path.split(self.scan_path)
        if path_split[0]:
            self.input_dir = path_split[0]
        else:
            self.input_dir = "."
        self.scan_file = path_split[1]

        i = self.scan_file.rfind('.nii')
        if i <= 0:
            i = self.scan_file.rfind('.tif')
        if i <= 0:
            print(f'error: input DTI scan must be either a "tif" or "nii" file: {self.scan_path}')
            sys.exit(-1)

        self.scan_path = os.path.join(self.input_dir, self.scan_file)

        if not os.path.isfile(self.scan_path):
            print(f"\nerror: input image file does not exist ({self.scan_path})\n")
            sys.exit(-1)

        # Check that the classifier method is valid, or that it is an exclusion file
        if self.classifier_method in self.ClassifierMethods:
            self.classifier_method = self.method_dict[self.classifier_method]
        else:
            self.rejected_frames_file = self.classifier_method
            self.classifier_method = ''
            if not os.path.isfile(self.rejected_frames_file):
                print(f"\nerror: classifier method is neither a recognized method, nor a frame exclusion file: ({self.rejected_frames_file})\n")
                sys.exit(-1)


    def params_message(self):
        ''' Returns string for display, listing all parameter values. '''
        cumulative_string = os.linesep
        cumulative_string += f"mmar version {Version().str()}{os.linesep}"
        date_string = datetime.today().strftime('%m/%d/%Y %H:%M')
        cumulative_string += f"    date run: {date_string}{os.linesep}"
        cumulative_string += f"    scan = {self.scan_file}{os.linesep}"
        cumulative_string += f"    input_dir = {self.input_dir}{os.linesep}"
        cumulative_string += f"    output_dir = {self.output_dir}{os.linesep}"
        if not self.ignore_meta:
            cumulative_string += f"    bval_file = {self.bval_file}{os.linesep}"
            cumulative_string += f"    bvec_file = {self.bvec_file}{os.linesep}"
            cumulative_string += f"    bvals_threshold = {self.bval_thresh}{os.linesep}"
        cumulative_string += f"    method = {self.classifier_method}{os.linesep}"
        if self.rejected_frames_file:
            cumulative_string += f"    rejected_frames = {self.rejected_frames_file}{os.linesep}"
        return cumulative_string
