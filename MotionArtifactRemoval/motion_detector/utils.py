'''
Copyright The Jackson Laboratory, 2022, 2023
authors: Jim Peterson, Abed Ghanbari
'''
import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt

from ..mmar import bad_frames


def static_feature_extractor(image, mask):
    '''
    Extracting static features for each mouse

    Parameters
    ----------
    image : numpy.ndarray
        2D array of the image
    mask : numpy.ndarray
        2D array of the mask

    Returns
    -------
    features : list
        List of the features
    '''
    features = [
        np.mean(image),
        np.std(image),

        np.mean(image[mask]),
        np.std(image[mask]),
        np.mean(image[mask]) / np.std(image[mask]),

        np.mean(image[mask==False]), #
        np.std(image[mask==False]),
        np.mean(image[mask==False]) / np.std(image[mask==False]),

        estimate_sigma(image), #

        psnr(image, image*mask),
        psnr(image, image*(1-mask))
        ]
    return features

def extract_feature_from_data(img_data):
    '''
    Calculates static and dynamic features from data
    Returns
    -------
    feature_mouse : numpy.ndarray
        Array of shape (nframes, nslices, nfeatures) containing the features for the
        entire mouse.
    '''
    nframes, nslices, nx, ny = img_data.shape
    nf_static = 11 # number of static features
    nf_dynamic = 7 # number of dynamic features -- has some information beyond 2D image
    feature_mouse = np.empty((nframes, nslices, nf_static+nf_dynamic))
    for j in range(nslices):
        mask = bad_frames.create_mask(img_data[0,j,:,:], mask_is_fg=False)[0]
        for i in range(nframes):
            feature_mouse[i,j,:nf_static] = static_feature_extractor(img_data[i,j,:,:], mask)

    # add general slice feature to each individual frame // based on Median Absloute Deviation
    for j in range(nslices):
        # calculate slice-wise features only for frames except the first one (b0)
        allframes_data_median = np.median(feature_mouse[1:,j,0])
        allframes_fg_data_median = np.median(feature_mouse[1:,j,2])
        allframes_bg_data_median = np.median(feature_mouse[1:,j,4])
        allframes_MAD = np.median([np.abs(feature_mouse[1:,j,0] - allframes_data_median)])
        allframes_fg_MAD = np.median([np.abs(feature_mouse[1:,j,2] - allframes_fg_data_median)])
        allframes_bg_MAD = np.median([np.abs(feature_mouse[1:,j,4] - allframes_bg_data_median)])
        for i in range(nframes):
            feature_mouse[i,j,nf_static] = allframes_fg_data_median
            feature_mouse[i,j,nf_static+1] = allframes_fg_MAD
            feature_mouse[i,j,nf_static+2] = allframes_bg_data_median
            feature_mouse[i,j,nf_static+3] = allframes_bg_MAD
            feature_mouse[i,j,nf_static+4] = allframes_MAD
            feature_mouse[i,j,nf_static+5] = allframes_data_median
            feature_mouse[i,j,nf_static+6] = j
    return feature_mouse


def psnr(img1, img2):
    '''
    Computes the Peak Signal to Noise Ratio (PSNR) between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        The first image.
    img2 : numpy.ndarray
        The second image.

    Returns
    -------
    float
        The PSNR between the two images.

    Notes
    -----
    The PSNR is computed as:

    .. math::
        PSNR = 20 \\cdot \\log_{10}(MAX_I) - 10 \\cdot \\log_{10}(MSE)

    where :math:`MAX_I` is the maximum possible pixel value of the image (255 for
    8-bit images) and :math:`MSE` is the mean-squared error between the two
    images.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    .. [2] https://www.mathworks.com/help/vision/ref/psnr.html
    .. [3] https://dsp.stackexchange.com/questions/38065/peak-signal-to-noise-ratio-psnr-in-python-for-an-image

    '''
    mse = np.mean((img1 - img2) ** 2) + sys.float_info.epsilon
    return 20*np.log10(np.max(img1) / np.sqrt(mse))

def find_excel_file(folder_path):
    '''
    This function finds all the excel files in a given folder.

    Parameters:
        folder_path (str): The path to the folder in which the function searches for excel files.

    Returns:
        excel_file_list (list): A list of all the excel files in the given folder.
    '''
    file_list = os.listdir(folder_path)
    excel_file_list = []
    for file in file_list:
        if file.endswith(".xlsx") and not file.startswith("~"):
            excel_file_list.append(file)
    return excel_file_list

def find_rejected_frames_from_csv(csv_file):
    '''
    Read lines of csv file with integer values
    One line per slice.  Each line is a comma separated list of rejected frames.
    The first column is the slice index.  Note: the slices must be in order, as the
    slice index is simply stripped off.
    '''
    with open(csv_file, 'r') as file:
        lines = file.read().splitlines()
    lines = [line.strip() for line in lines]
    lines = [line.split(',') for line in lines]
    lines = [line[1:] for line in lines]
    lines = [[int(l) for l in line if l] for line in lines ]
    lines = [(i, line) for i, line in enumerate(lines)]

    return lines

def write_rejected_frames_to_csv(csv_file, rejected_frames):
    '''
    This function writes the rejected frames to a csv file.
    '''
    with open(csv_file, 'w') as file:
        for frame in rejected_frames:
            if isinstance(frame, int):
                file.write(str(frame))
            else:
                for i in frame:
                    file.write(str(i) + ',')
            file.write('\n')

def make_csv_from_excel(parent_folder):
    '''
    This function creates a csv file from the excel file of JAX annotations
    '''
    mouse_list = os.listdir(parent_folder)
    print(mouse_list)
    for mouse in mouse_list:
        if mouse.startswith("CK_DIF"):
            arr = get_rejected_frames_mouse(parent_folder, mouse)
            csv_file = os.path.join(parent_folder, mouse + '_rejected_frames.csv')
            write_rejected_frames_to_csv(csv_file, arr)

def fill_rejected_list(rejected_list, nslices):
    '''
    This function fills the list of rejected lists with empty lists if there are missing slices.
    '''
    return rejected_list.extend([[]]*(nslices-len(rejected_list)))

def shift_rejected_list(rejected_list, shift):
    '''
    This function shifts the values for each slice in the list of rejected frames.
    '''
    for i in range(len(rejected_list)):
        line = [x-shift for x in rejected_list[i][1]]
        rejected_list[i] = (i, line)
    return rejected_list

def get_rejected_frames(parent_folder, mouse_list):
    '''
        Parameters
        ----------
        parent_folder : str
            The path to the folder containing the data for all the mice.
        mouse_list : list of str
            The names of the mice to be analyzed.

        Returns
        -------
        rejected : dict of lists
    '''
    rejected = {}
    for mouse in mouse_list:
        rejected[mouse] = get_rejected_frames_mouse(parent_folder, mouse)
    return rejected

def get_rejected_frames_mouse_set(parent_folder, mouse, zero_indexed=True):
    rej_list = get_rejected_frames_mouse(parent_folder, mouse, zero_indexed=zero_indexed)
    rej_set = [set() for i in range(len(rej_list))]
    for i in range(len(rej_list)):
        if isinstance(rej_list[i], int):
            rej_set[i] = set([rej_list[i]+2])
        else:
            rej_set[i] = set([i+2 for i in rej_list[i]])
    return rej_set

def get_rejected_frames_mouse(parent_folder, mouse, zero_indexed=True):
    '''
    Parameters
    ----------
    parent_folder : str
        The parent folder where the excel file is located
    mouse : str
        The mouse name
    zero_indexed : bool, optional
        Whether the returned rejected frame list is zero indexed or not, by default True

    Returns
    -------
    rejected : list
        A list of lists of rejected frames for each trial

    '''
    rejected_frames_file = find_excel_file(os.path.join(parent_folder, mouse))
    if len(rejected_frames_file)==1:
        df = pd.read_excel(os.path.join(parent_folder, mouse, rejected_frames_file[0]), engine='openpyxl')
        rejected = [[]]*17
        for i, val in enumerate(df[df.columns[0]]):
            if isinstance(val, int):
                if isinstance(df.iloc[i,1], str):
                    if zero_indexed:
                        rejected[val-1]=[int(i)-2 for i in df.iloc[i,1].split(',') if len(i)>0]
                    else:
                        rejected[val-1]=[int(i)-1 for i in df.iloc[i,1].split(',') if len(i)>0]

                elif isinstance(df.iloc[i,1], int):
                    if zero_indexed:
                        rejected[val-1]=df.iloc[i,1]-2
                    else:
                        rejected[val-1]=df.iloc[i,1]-1
        return rejected
    if len(rejected_frames_file)>1:
        raise ValueError(f"More than one excel file found: \n{rejected_frames_file}")
    else:
        return [[np.nan]]*17


def set_to_one(A, B):
    for i in range(len(A)):
        if A[i] != []:
            if isinstance(A[i], int):
                B[A[i]][i] = 1
                continue
            for j in A[i]:
                B[j][i] = 1
    return B

def normalize_data(data, mode='zscore'):
    '''
    This function normalizes the data in the data array
    '''
    if mode == 'zscore':
        data = (data - np.mean(data)) / np.std(data)
    elif mode == 'minmax':
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif mode == 'mad':
        data = (data - np.median(data)) / np.median(np.abs(data - np.median(data)))
    return data

def get_X(features):
    '''
    This function returns the X matrix from the features dictionary
    '''
    X = []
    for key in features:
        X.append(features[key])
    X = np.array(X)

    return X

def find_max_frames(rej_list):
    max_frames = 0
    for csv_file in rej_list:
        for vals in rej_list[csv_file]:
            if vals[1]:
                curr_max = max(vals[1])
                if curr_max > max_frames:
                    max_frames = curr_max
    return max_frames

def show_example_frames(X_attrib, Y, mouse_list, start_idx, n_examples=10):
    '''
    This function shows some examples of rejected frames and accepted frames.
    '''
    # randomly select n_examples from the data with Y=1
    rejected_indices = np.random.choice(np.where(Y==1)[0], n_examples, replace=False)
    accepted_indices = np.random.choice(np.where(Y==0)[0], n_examples, replace=False)

    # plot the examples
    _, ax = plt.subplots(2, n_examples, figsize=(5,6))
    for i in range(n_examples):

        i_mouse, i_frame, i_slice = X_attrib[rejected_indices[i],...]
        img = sitk.GetArrayFromImage(sitk.ReadImage(mouse_list[i_mouse]))[start_idx+i_frame,i_slice,:,:]
        ax[0,i].imshow(img, cmap='gray')
        ax[0,i].set_title(f"mouse: {i_mouse+1} frame: {start_idx+i_frame+1} slice: {i_slice+1}")

        i_mouse, i_frame, i_slice = X_attrib[accepted_indices[i],...]
        img = sitk.GetArrayFromImage(sitk.ReadImage(mouse_list[i_mouse]))[start_idx+i_frame,i_slice,:,:]
        ax[1,i].imshow(img, cmap='gray')
        ax[1,i].set_title(f"mouse: {i_mouse+1} frame: {start_idx+i_frame+1} slice: {i_slice+1}")
    plt.show()
