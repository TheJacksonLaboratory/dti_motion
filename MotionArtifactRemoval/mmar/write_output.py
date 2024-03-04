'''
Copyright The Jackson Laboratory, 2022
authors: Jim Peterson, Abed Ghanbari

Implemetation of functions for writing results
'''

import os
import numpy as np
import SimpleITK as sitk
import tifffile as tiff # we might be able to save tif files using only sitk
import nibabel as nib


def write_edited_data(output_dir, src_img_path, img, tensor_metadata,
    rejected_frames_by_slice, use_subdirs, save3d, log=None):
    '''
    Write all output files

    Parameters:
        output_dir : string
            directory for output.  Will be created if it does not exist
        src_img_path : string
            original image path
        img : 4D image
            original image
        tensor_metadata : TensorMetadata class instance
            contents of bval and bvec files and methods for manipulation
        rejected_frames_by_slice : list of lists
            rejected frames - a list per slice
        use_subdirs : boolean
            when true a subdirectory is created for each slice
        save3d : boolean
            when true, slice is saved as a multi-frame 2D image,
            otherwise all slices are saved in each slice file
        log : log object
            used for logging messages

    Return:
        error string or empty string if successful
    '''

    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir)
        except OSError as error:
            error_str = f"Could not create output directory: {output_dir} ({error})"
            if log:
                log.add_msg(error_str)
            return error_str

    if log:
        log.add_msg(f"{os.linesep}writing modified slices:")

    img_file = os.path.split(src_img_path)[1]

    #
    # Write each slice in a new group
    #
    for z_idx, rejected_frames in enumerate(rejected_frames_by_slice):
        slice_grp = "Z" + str(z_idx).zfill(3)
        if use_subdirs:
            slice_path = os.path.join(output_dir, slice_grp)
            if not os.path.isdir(slice_path):
                try:
                    os.mkdir(slice_path)
                except OSError as error:
                    error_str = f"Could not create directory for edited slice: {slice_path} ({error})"
                    if log:
                        log.add_msg(error_str)
                    return error_str
            prefix = ''
        else:
            slice_path = output_dir
            prefix = slice_grp+'_'

        tensor_metadata.write(slice_path, prefix, rejected_frames)

        img_path = os.path.join(slice_path, prefix+img_file)
        if log:
            log.add_msg(f"    slice {z_idx}: {img_path}")
        if save3d:
            write_img_slice_3d(img, z_idx, rejected_frames, img_path)
        else:
            write_img_slice_4d(img, src_img_path, rejected_frames, img_path)

    #
    # Write the rejected frames file
    #
    reject_frames_path = os.path.join(output_dir, "excluded_frames.csv")
    write_rejected_frames_file(reject_frames_path, rejected_frames_by_slice)

    return ""

def write_edited_slice(output_dir, src_img_path, img, tensor_metadata,
    rejected_frames, slice_number, use_subdirs, save3d, log=None):
    '''
    Write all output files

    Parameters:
        output_dir : string
            directory for output.  Will be created if it does not exist
        src_img_path : string
            original image path
        img : 4D image
            original image
        tensor_metadata : TensorMetadata class instance
            contents of bval and bvec files and methods for manipulation
        rejected_frames : list of lists
            rejected frames - a list per slice
        slice_number: integer
            zero-based slice number
        use_subdirs : boolean
            when true a subdirectory is created for each slice
        save3d : boolean
            when true, slice is saved as a multi-frame 2D image,
            otherwise all slices are saved in each slice file
        log : log object
            used for logging messages

    Return:
        error string or empty string if successful
    '''

    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir)
        except OSError as error:
            error_str = f"Could not create output directory: {output_dir} ({error})"
            if log:
                log.add_msg(error_str)
            return error_str

    if log:
        log.add_msg(f"{os.linesep}writing modified slices:")

    img_file = os.path.split(src_img_path)[1]

    #
    # Write the slice
    #
    slice_grp = "Z" + str(slice_number).zfill(3)
    if use_subdirs:
        slice_path = os.path.join(output_dir, slice_grp)
        if not os.path.isdir(slice_path):
            try:
                os.mkdir(slice_path)
            except OSError as error:
                error_str = f"Could not create directory for edited slice: {slice_path} ({error})"
                if log:
                    log.add_msg(error_str)
                return error_str
        prefix = ''
    else:
        slice_path = output_dir
        prefix = slice_grp+'_'

    tensor_metadata.write(slice_path, prefix, rejected_frames)

    img_path = os.path.join(slice_path, prefix+img_file)
    if log:
        log.add_msg(f"    slice {slice_number}: {img_path}")
    if save3d:
        write_img_slice_3d(img, slice_number, rejected_frames, img_path)
    else:
        write_img_slice_4d(img, src_img_path, rejected_frames, img_path)

    return ""


def write_img_slice_4d(img_4d, src_img_path, rejected_frames, dst_img_path):
    '''
    Write one slice of the 4D image as a 3D image, removing the specified frames

    Parameters:
        img_4d : 4D image
            original scan
        src_img_path : string
            path to the original scan
        rejected_frames : list
            indices of the frames to remove
        dst_img_path : string
            path to the new file
    Return:
        None
    '''
    img = img_4d.copy()
    if len(rejected_frames):
        img = np.delete(img, rejected_frames, 0)
    img = np.transpose(img, (3,2,1,0))
    nii_img = nib.load(src_img_path)
    nii_img_data = nib.Nifti1Image(img, nii_img.affine, nii_img.header)
    nii_img_data.to_filename(dst_img_path)


def write_img_slice_3d(img_4d, slice_number, rejected_frames, dst_img_path):
    '''
    Write one slice of the 4D image as a 3D image, removing the specified frames

    Parameters:
        img_4d : 4D image
            original scan
        slice_number : integer
            z-index of the slice
        rejected_frames : list
            indices of the frames to remove
        dst_img_path : string
            path to the new file
    Return:
        None
    '''
    img = img_4d[:, slice_number, :, :]
    if len(rejected_frames):
        img = np.delete(img, rejected_frames, 0)

    if dst_img_path.endswith("nii.gz") or dst_img_path.endswith("nii"):
        img_itk = sitk.GetImageFromArray(img)
        sitk.WriteImage(img_itk, dst_img_path)
    else:
        tiff.imsave(dst_img_path, img)


def write_rejected_frames_file(output_file_path, rejected_frames_by_slice):
    '''
    Write the list of rejected frames.

    Parameters:
        output_dir : str
            The path to the directory in which to write the file.
        rejected_frames_by_slice : list
            A list of lists of rejected frames.  Each list is a list of
    Return:
        None
    '''
    body = "z-index, excluded_frames\n"

    for idx, frame_list in enumerate(rejected_frames_by_slice):
        body += f"{idx}"
        for frame in frame_list:
            body += f", {frame}"
        body += "\n"

    with open(output_file_path, 'w', encoding="latin-1") as file:
        file.write(body)

def write_slice_image_as_tiff(path, z_slice):
    '''
    Write a slice to a file.

    Parameters:
        path : str
            The path to the file to write.
        z_slice : MultiFrameSlice
            The slice to write.

    Returns:
        None
    '''
    img = z_slice['img']
    if z_slice['rejected']:
        img = np.delete(img, z_slice['rejected'], 0)
    tiff.imsave(path, img)


def write_slice_image_as_nii(path, z_slice):
    '''
    Write a slice to a file.

    Parameters:
        path : str
            The path to the file to write.
        z_slice : MultiFrameSlice
            The slice to write.

    Returns:
        None
    '''
    img = z_slice['img']
    if z_slice['rejected']:
        img = np.delete(img, z_slice['rejected'], 0)
    img_itk = sitk.GetImageFromArray(img)
    sitk.WriteImage(img_itk, path)
