# Copyright The Jackson Laboratory, 2022
# authors: Jim Peterson, Abed Ghanbari
'''
Functions for determination of bad frames in a given slice.
'''

import pickle
import numpy as np
from ..motion_detector import utils as mdu
from .create_mask import create_mask


def find_bad_frames_mad(img, fg_thresh, bg_thresh):
    '''
    Identify bad frames using the MAD method.

    Parameters:
        img : 4D image
            original image
        fg_thresh : float
            foreground threshold used by MAD
        bg_thresh : float
            background threshold used by MAD

    Return:
        list of lists.  Each element is the list of rejected frames for a slice
    '''

    frames_by_slice = []
    _, slice_count, _, _ = img.shape
    for z in range(slice_count):
        slice_img = img[:, z, :, :]
        mask, _ = create_mask(slice_img[0,:,:], mask_is_fg=False)
        profile_foreground, profile_background = calculate_profiles(slice_img, mask)
        e_fg = get_outliers(profile_background, mode='MAD', mad_thr=fg_thresh)
        e_bg = get_outliers(profile_foreground, mode='MAD', mad_thr=bg_thresh)
        for n in e_bg:
            if not n in e_fg:
                e_fg.append(n)
        rejected_frames = sorted(e_fg, reverse=False)
        frames_by_slice.append(rejected_frames)
    return frames_by_slice


def find_bad_frames_ml(img,
                        model_path=None,
                        cv_result=None,
                        classifier='Logistic Regression',
                        norm_mode='mad',
                        num_slices=None,
                        num_frames_large=None,
                        prob_thresh=0.5,
                        prob_thresh_exceptions=[]):
    '''
    Finds the bad frames in a slice using machine learning.
    '''
    if num_slices is None:
        num_slices = img.shape[1]
    if num_frames_large is None:
        num_frames_large = img.shape[0]
    assert img.shape[0] < num_frames_large # we should only include large frame without b0 frame

    prob_thresh_list = [prob_thresh]*num_slices
    for ex in prob_thresh_exceptions:
        prob_thresh_list[ex[0]] = ex[1]

    # load model
    if cv_result is None:
        cv_result = pickle.load(open(model_path, "rb"))

    img = img.astype('uint16')

    # find all zero slices
    non_zero_idx = np.where(np.sum(img, axis=(0,2,3)))[0]

    feature = mdu.extract_feature_from_data(img[:,non_zero_idx,:,:])
    for i in range(feature.shape[2]): # for individual feature
        feature[:,:,i] = mdu.normalize_data(feature[:,:,i], mode=norm_mode)

    # feature size: (n_frames, nslices, n_features)
    X = feature.reshape(-1,feature.shape[-1])
    nan_idx = np.isnan(np.sum(X,axis=1))
    # predict
    Y_probability = np.zeros(X.shape[0])
    Y_probability[~nan_idx] = cv_result[classifier][0].predict_proba(X[~nan_idx,:])[:,1]
    Y_probability_reshaped = np.zeros((img.shape[0], num_slices))
    Y_probability_reshaped[:,non_zero_idx] = Y_probability.reshape((-1, len(non_zero_idx)))

    frames_by_slice = []
    probs_by_slice = []
    for s_idx in range(num_slices):
        frame_list = []
        prob_list = []
        for f_idx in range(num_frames_large-1):
            prob = Y_probability_reshaped[f_idx, s_idx]
            if prob >= prob_thresh_list[s_idx]:
                frame_list.append(f_idx)
            prob_list.append(prob)
        frames_by_slice.append(frame_list)
        probs_by_slice.append(prob_list)

    return  frames_by_slice, probs_by_slice


def calculate_profiles(image_stack, mask):
    '''
    Calculate profiles from a stack of images.

    Parameters
    ----------
    image_stack : numpy.ndarray
        A stack of images.
    mask : numpy.ndarray
        A mask for the image stack.

    Returns
    -------
    profile_foreground : numpy.ndarray
        A profile of the foreground.
    profile_background : numpy.ndarray
        A profile of the background.
    '''
    nframes = image_stack.shape[0]
    profile_foreground = np.zeros(nframes)
    profile_background = np.zeros(nframes)
    for i_frame in range(nframes):
        profile_foreground[i_frame] = np.sum(image_stack[i_frame,:,:] * (mask)) / np.sum((mask))
        profile_background[i_frame] = np.sum(image_stack[i_frame,:,:] * (1-mask)) / np.sum((1-mask))
    return profile_foreground, profile_background


def get_outliers(profile, mode='MAD', ratio=None, mad_thr=None):
    '''
    This function identifies outliers in a given dataset.

    Parameters
    ----------
    data : list
        A list of numbers.
    mode : str, optional
        The method used to identify outliers.
        'std' uses standard deviation method.
        'MAD' uses median absolute deviation method.
        The default is 'MAD'.
    ratio : float, optional
        The ratio used to identify outliers.
        The default is None.
    mad_thr : float, optional
        The threshold used to identify outliers using median absolute deviation method.
        The default is None, however in
        [https://www.sciencedirect.com/science/article/abs/pii/S0022103113000668]
        it has been suggested to use 2.5

    Returns
    -------
    outliers : list
        A list of indices of outliers in the given dataset.

    '''
    if mode=='std':
        mean = np.mean(profile[1:])
        std = np.std(profile[1:])
        outliers = []
        for i in range(1,len(profile)):
            if profile[i] > mean + ratio*std or profile[i] < mean - ratio*std:
                outliers.append(i)
    elif mode=='MAD':
        # calculate median and median absolute deviation
        # ref: https://www.sciencedirect.com/science/article/abs/pii/S0022103113000668
        profile_median = np.median(profile[1:])
        profile_mad = np.median([np.abs(profile[1:] - profile_median)])
        # identify outliers
        cut_off = profile_mad * mad_thr
        lower, upper = profile_median - cut_off, profile_median + cut_off
        outliers = []
        for i in range(1,len(profile)):
            if profile[i] < lower or profile[i] > upper:
                outliers.append(i)
    return outliers
