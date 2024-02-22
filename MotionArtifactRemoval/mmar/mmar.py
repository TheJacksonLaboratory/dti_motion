# Copyright The Jackson Laboratory, 2022
# authors: Jim Peterson, Abed Ghanbari

'''
This script removes frames affected by motion in DTI scans.
To run after instalation of the module:
    python -m MotionArtifactRemoval.mmar.mmar <image-file> <optional-arguments>

To Do:
    - catch possible exception when bval and bvec file entries are converted to numbers
    - catch exceptions when writing output files
'''
import sys
import os
import csv
import SimpleITK as sitk

from .bad_frames import find_bad_frames_ml, find_bad_frames_mad
from .write_output import write_edited_data
from .command_line_params import CommandLineParams
from .message_log import MessageLog
from .tensor_metadata import TensorMetadata


def main():
    '''
    This is the main entry point for the program.  It does the following:
        1. parse command line arguments
        2. open input image data and metadata
        3. separate the scans, when more than one is present
        4. compute the rejected frames for each scan
        5. write the output files, with rejected frames removed
    Return
        0 if successful, -1 otherwise
    '''

    #
    # Parse the command line parameters
    #
    params = CommandLineParams()
    params.check_parameters()
    log = MessageLog(print)
    log.add_msg(params.params_message())

    #
    # Open the input files.
    # After this block we will use:
    #    img - the 4D scan
    #    bval_list - list of tensor magnitudes per frame/direction
    #    bvec_list - list of tensor vector values per frame/direction
    #    frame_starts - list of 0-diffusion frames (e.g. [0, 8])
    img = sitk.GetArrayFromImage(sitk.ReadImage(params.scan_path))
    frames, slices, height, width = img.shape
    bval_frames = 0
    if params.ignore_meta:
        bval_frames = frames

    tensor_metadata = TensorMetadata(log)
    if bval_frames > 0:
        tensor_metadata.ignore_metadata(bval_frames)
    elif params.bval_file and params.bvec_file:
        tensor_metadata.use_bval_bvec_files(params.bval_file, params.bvec_file)
    else:
        tensor_metadata.find_files_with_default_names(params.scan_path)

    #
    # print the image geometry
    #
    log.add_msg(f"    sequences = {tensor_metadata.scan_count()}")
    log.add_msg(f"    width = {width}")
    log.add_msg(f"    height = {height}")
    log.add_msg(f"    slices = {slices}")
    log.add_msg(f"    frames = {frames}")
    if (not params.ignore_meta) \
        and (not ((frames == len(tensor_metadata.bval_list)) and (frames == len(tensor_metadata.bvec_list)))):
        log.add_msg("error: input files do not match in frame count")
        sys.exit(-1)
    large_shell_start, large_shell_len = tensor_metadata.get_longest_scan()
    log.add_msg(f"    large-shell frame range: [{large_shell_start}, {large_shell_start+large_shell_len-1}]")
    if params.ml_model:
        log.add_msg(f"    model file: {params.ml_model}")

    if (params.classifier_method) and (not params.classifier_method=='MAD'):  # method is one of the ML methods
        if params.ml_model:
            model_path = params.ml_model
        else:
            model_path = os.path.join(nth_parent_dir(__file__, 2),'motion_detector','trained_models','frame_rejection_all_80.pkl')
        if not os.path.isfile(model_path):
            print(f"\nerror: model file does not exist ({model_path})\n")
            sys.exit(-1)

        log.add_msg(" ")
        log.add_msg(f"    classifier key: {params.classifier_method}")
        log.add_msg(f"    model_path: {model_path}")
        log.add_msg(f"    prob_thresh: {params.ml_prob_thresh}")
        if params.ml_prob_thresh_exceptions:
            log.add_msg(f"    ml_prob_thresh_exceptions: {params.ml_prob_thresh_exceptions}")

        rejected_frames, _ = find_bad_frames_ml(
            img[large_shell_start+1:large_shell_start+large_shell_len,...],
            model_path,
            None,
            params.classifier_method,
            num_slices=slices,
            num_frames_large=large_shell_len,
            prob_thresh=params.ml_prob_thresh,
            prob_thresh_exceptions=params.ml_prob_thresh_exceptions)
        rejected_frames = [[x+large_shell_start+1 for x in frames] for frames in rejected_frames]

    elif params.classifier_method=='MAD':
        log.add_msg(" ")
        log.add_msg(f"    classifier key: {params.classifier_method}")
        log.add_msg(f"    fg_thresh = {params.fg_thresh}")
        log.add_msg(f"    bg_thresh = {params.bg_thresh}")
        rejected_frames = find_bad_frames_mad(img[large_shell_start+1:,...], params.fg_thresh, params.bg_thresh)
        rejected_frames = [[x+large_shell_start+1 for x in frames] for frames in rejected_frames]

    else: # rejected frames are read from a file
        if not os.path.isfile(params.rejected_frames_file):
            log.add_msg(f"rejected frames file does not exist: {params.rejected_frames_file}")
            sys.exit(-1)
        lines = []
        with open(params.rejected_frames_file, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            lines = list(reader)
        rejected_frames = [[] for _ in range(slices)]

        for line in lines:
            if (len(line)==0) or (not line[0].isnumeric()):
                continue
            z_idx = int(line[0])
            if z_idx<0 or z_idx>=slices:
                continue
            frames = [int(i) for i in line[1:]]
            rejected_frames[z_idx] = frames

    #
    # Tabulate maximum angle between good frames
    #     JGP - this should use the info in the bvec file.
    #     As it is, it assumes that the frames are equally spaced and fill the full 360 deg.
    max_angles = []
    degree_per_frame = 360.0/(large_shell_len)
    for frames in rejected_frames:
        last_frame = -1
        longest_chain = 1
        current_chain_length = 1
        for frame in frames:
            if frame == (last_frame + 1):
                current_chain_length += 1
                if current_chain_length > longest_chain:
                    longest_chain = current_chain_length
            else:
                current_chain_length = 1
            last_frame = frame
        max_angles.append(longest_chain*degree_per_frame)

    #
    # Report rejected frames
    #
    log.add_msg("\nrejected frames: slice: good_frames, max_void_degree, rejected_frames")
    for i in range(slices):
        log.add_msg(f"    Slice {i}: {large_shell_len-len(rejected_frames[i])}/{large_shell_len}, {round(max_angles[i], 1)}, {rejected_frames[i]}")

    #
    # Write output files
    #
    write_edited_data(params.output_dir, params.scan_path, img, tensor_metadata, rejected_frames, params.use_subdirs, params.save3d, log)
    log_file_path = os.path.join(params.output_dir, "log.txt")
    log.save_log(log_file_path)

    sys.exit(0)


def nth_parent_dir(path, count):
    '''
    Generate the nth parent directory of the path object.

    Parameters:
        path : string
            The path to interpert
        count : integer
            Which parent directory to return
    Return : string
        nth parent directory
    '''
    dir_tmp = path
    for _ in range(count):
        dir_tmp = os.path.dirname(dir_tmp)

    return dir_tmp


if __name__== "__main__":
    main()
