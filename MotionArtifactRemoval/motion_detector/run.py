'''
 Copyright The Jackson Laboratory, 2022
 authors: Jim Peterson, Abed Ghanbari

 This script creates a trained model for classifying frames based
 on in put annotation data.  The resulting model file (.pkl) us used
 by the mmar programs.

'''

import sys
import argparse
import csv
import pickle
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from .train import MotionFrameDetector, ClassifierNames
from .utils import ( find_rejected_frames_from_csv,
                     shift_rejected_list,
                     find_max_frames,
                     extract_feature_from_data,
                     show_example_frames)

def train_model(args):
    '''
    Perform the training based on the annotated data.
    '''
    anno_list = args.anno_list
    output_model_path = args.output_model_path
    classifier_names = args.classifier_names
    search_param = False
    start_idx = args.start_idx
    shift_csv_labels = args.start_idx + int(args.one_indexed)

    if not anno_list:
        print("Error: annotation list must not be empty")
        return False

    mouse_list = []
    csv_list =  []
    with open(anno_list, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        lines = list(reader)
        lines = [line for line in lines if line[0][:1] != '#' ]
        mouse_list = [line[1].strip() for line in lines]
        csv_list = [line[0].strip() for line in lines]

    if len(csv_list) != len(mouse_list):
        print("Error: mouse list and csv list must be the same length.")
        return False

    # read all rejected frames from csv file
    rej_list = {}
    for csv_file in csv_list:
        rej_list[csv_file] = find_rejected_frames_from_csv(csv_file)

    if args.debug:
        print('rejected frames:')
        for csv_file in csv_list:
            print(csv_file, ':\n', rej_list[csv_file])

    if shift_csv_labels:
        for csv_file in csv_list:
            rej_list[csv_file] = shift_rejected_list(rej_list[csv_file], shift_csv_labels)

    if args.debug:
        print("rejected frames after shifting:")
        for csv_file in csv_list:
            print(csv_file, ':\n', rej_list[csv_file])

    # read mouse files
    _max = find_max_frames(rej_list) # max frame number included in csv files

    # remove all mice that have less than min_rejected_frames
    for i in range(len(mouse_list)-1, -1, -1):
        csv_file = csv_list[i]
        sum_rejected_frames = sum([len(el[1]) for el in rej_list[csv_file]])
        #print(f"rejected_count, {csv_file}, {sum_rejected_frames}")
        if sum_rejected_frames < args.min_rejected_frames:
            print(f'removing {csv_file} because it has less than {args.min_rejected_frames} rejected frames')
            csv_list.remove(csv_file)
            mouse_list.remove(mouse_list[i])
    for i, mouse_name in enumerate(mouse_list):
        img = sitk.GetArrayFromImage(sitk.ReadImage(mouse_name))[start_idx:,:,:,:]
        assert _max <= img.shape[0], f"Max frames in rej_list: {_max} is larger than the number of frames in mouse {mouse_name}:{img.shape[0]}"

        # sum along all except first dimension, so we won't calculate features for empty slices
        zero_slice_idx = img.sum(axis=(0,2,3)) != 0
        features = extract_feature_from_data(img[:,np.where(zero_slice_idx)[0],:,:])
        if i==0: # we create X, Y placeholder here since we need number of features
            # place holder for Y and zero_slice_idx
            Y = np.zeros([len(mouse_list), img.shape[0], img.shape[1]]) # mice, frames, slices
            X = np.zeros([len(mouse_list), img.shape[0], img.shape[1], features.shape[-1]]) # mice, frames, slices, features
            x_attrib = np.indices([len(mouse_list), img.shape[0], img.shape[1]])
            x_attrib = np.moveaxis(x_attrib, 0, -1)

        if len(rej_list[csv_list[i]]) > img.shape[1]:
            print(f"{csv_list[i]}: Number of records ({len(rej_list[csv_list[i]])}) exceeds the number of slices ({img.shape[1]}) in the image.")
            return False
        for rej in rej_list[csv_list[i]]:
            Y[i,rej[1],rej[0]] = 1

        X[i, :, np.where(zero_slice_idx)[0], :] = np.swapaxes(features, 0, 1)

    # reshape X and Y so each row represnts a single frame
    Y = Y.reshape(-1,1)
    X = X.reshape(-1,X.shape[-1])
    x_attrib = x_attrib.reshape(-1,3)

    # find frames that all features are zero (an indicitive of empty/zero images)
    # and remove them for training
    zero_feature = X.sum(axis=1)!=0
    Y = Y[zero_feature,:]
    X = X[zero_feature,:]
    x_attrib = x_attrib[zero_feature,:]

    nan_features_idx = ~np.isnan(X.sum(1))
    X = X[nan_features_idx,:]
    Y = Y[nan_features_idx,0]
    x_attrib = x_attrib[nan_features_idx,:]

    # show example of data
    if args.show_examples:
        show_example_frames(x_attrib, Y, mouse_list, start_idx, n_examples=5)
    model = {}
    cv_results = {}
    for clf in classifier_names:
        model[clf] = MotionFrameDetector(
            X, Y,
            train_balanced=True,
            classifier_name=clf,
            search_param=search_param
            )

        cv_results[clf] = model[clf].train()

        # print the best parameters, if search_param is True then this is from
        # the grid search and if set to False then it is the best parameters from Parameters class
        print(f"{clf} Cross Validation results: {cv_results[clf][0].best_params_}")

        print(f"\nTraining stats for {clf}:")
        model[clf].print_results()
        if args.plot_results:
            model[clf].plot_roc_curves()

    if args.plot_results:
        plt.show()

    # save the model to output_model_path if output_model_path is specified
    if output_model_path is not None:
        with open(output_model_path, 'wb') as file:
            pickle.dump(cv_results, file)

    return True


def main():
    '''
        example:

        python -m MotionArtifactRemoval.motion_detector.run \
            --anno_list anno_list.csv \
            --output_model_path new_model.pkl \
            --classifier_names 'Logistic Regression' 'SVM' 'Random Forest' 'AdaBoosted decision tree' 'Gradient Boosted Decision Tree' 'Gaussian Naive Bayes' \
            --start_idx 8 \
            --one_indexed \
            --plot_results \
            --min_rejected_frames 0 \
        2>train.err 1>train.log

    '''
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--anno_list', type=str,
        help='CSV list of pairs of annotation files and corresponding image files.')
    parser.add_argument('--output_model_path', type=str,
        help='path to save the model')
    choices = [x.value for x in ClassifierNames]
    parser.add_argument('--classifier_names', nargs='+', type=str,
        help=f'list of classifiers to use, choices: {choices}', default=choices, metavar='clf')
    parser.add_argument('--start_idx', type=int, default=0,
        help='start index (0-based) of the frames to process')
    parser.add_argument('--one_indexed', action='store_true',
        help='annotation indexes are 1-based (otherwise, 0-based)')
    parser.add_argument('--plot_results', action='store_true',
        help='plot the ROC curves, saved as roc.png')
    parser.add_argument('--show_examples', action='store_true',
        help='show examples of data')
    parser.add_argument('--rand_seed', type=int, default=0,
        help="Seed for random number generator.  Set to -1 to eliminate seeding. (default=0)")
    parser.add_argument('--min_rejected_frames', type=int, default=0,
        help="Mice with fewer than this number of rejected frames are not used. (default=0)")
    parser.add_argument('--debug', action='store_true',
        help='print additional debug information')

    args = parser.parse_args()
    print(f"anno_list: {args.anno_list}")
    print(f"output_model_path: {args.output_model_path}")
    print(f"classifier_names: {args.classifier_names}")
    print(f"start_idx: {args.start_idx}")
    print(f"plot_results: {args.plot_results}")
    print(f"show_examples: {args.show_examples}")
    print(f"rand_seed: {args.rand_seed}")
    print("\n==========================================\n")

    if args.rand_seed >= 0:
        np.random.seed(args.rand_seed)

    if train_model(args):
        print("Training Done!")
        print(f"Output model saved at: {args.output_model_path}")
    else:
        print("Training Failed.")
        print("Model NOT saved.")
        sys.exit(-1)

    sys.exit(0)

if __name__== "__main__":
    main()
