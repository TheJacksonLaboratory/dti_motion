# MRI Motion Artifact Removal (mmar) Tools
**Authors:** _Jim Peterson (jim.peterson@jax.org)_, _Abed Ghanbari (abed.ghanbari@jax.org)_

This package contains tools for detection and removal of motion artifacts in diffusion-weighted
MRI images (DTI and NODDI).
The input files (nii, nii.gz, or tif) are 4D images consisting
of z-stacks of images, each of which is a multi-frame 2D image, with each frame representing
a particular direction of diffusion. Each image file is accompanied by "bvec" and "bval" files
that hold the tensor direction and magnitude respectivly for all frames.  The input files may
contain a single scan, or both "small-shell" and "large-shell" scans.
The start of a scan is signified by a zero-diffusion frame, which is identified by a near-zero
intensity in the bval file.

For each slice in the z-stack, frames affected by motion are removed. Since the affected frames
will be different for each slice, the final output is a set of image and metadata files for each slice.

There are three application in this package: a command-line application (CLI) that  is well suited for batch processing,
a graphical interface application (GUI) for inspection and validation, and a training application to create the models used for frame classification.
All three are Python-based applications that are installed as a single Python module.

repo: [https://github.com/TheJacksonLaboratory/dti_motion](https://github.com/TheJacksonLaboratory/dti_motion)

## Classification Methods
There are several machine learning methods of classification offered by both the command line and GUI applications.  Additionally, the CLI offers a simple threshold-based method using Median Absolute Deviation (MAD).

## Installation
MMAR can be installed in an Anaconda virtual environment following these steps.  If you do not have the Anaconda package manager the installer can be downloaded for [MacOS](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg)
or [Windows](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe), and install with:


```
curl -o ~/miniconda.sh 'https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh'
bash ~/miniconda.sh -b -p $HOME/miniconda
```

A virtual environment for this project can be created using:
```
conda create -n MMAR python=3.7.0 -y
conda activate MMAR
```

After cloning this repository into a directory (e.g. dti_motion), the MMAR package can be installed with the following commands.
```
cd dti_motion
pip install .
```



#### Apple Silicon Chip - M1
To install this package on newer Macbook you should first install Rossetta and then follow steps above.

 
> After installing Rosetta:

> `softwareupdate --install-rosetta --agree-to-license`

> It seems to be a good idea to run your entire Terminal in Rosetta:

> Go to Finder > Applications and find your Terminal (this can also be other terminal > app, like iTerm in my case)
> Right-Click the App and Duplicate it and rename it "Terminal i386"
> Right-Click "Terminal i386" > Get Info > Enable Open using Rosetta
> Click to Open the Terminal, type arch to verify it says i386 now.
> Right-Click the Terminal i386 in your Dock and click "Keep in Dock" for future use.
> It is important to install/update/deploy within the "Terminal i386" window now, your > normal Terminal will be arm64 and won't have the same libraries. Consider full > emulation as the easiest solution without a lot of workaround with flags and running > multiple brew in parallel. It just works.


## Testing the Installation
The installation can be tested by running the applications in the MMAR module after
activating the virtual environment.
Run the commands below to activate the environment, then start the command-line and GUI applications.
```
conda activate MMAR
python -m MotionArtifactRemoval.mmar.mmar -h
python -m MotionArtifactRemoval.mmar_gui.gui
```
Note the first time these applications are run it may take a minute or two for them to start.

## Command Line Application (*mmar*)
The command line application is well suited for batch processing samples, non-interactively.
After the virtual environment is activated, the **_mmar_** application can
be run with the command line:
```Bash
python -m MotionArtifactRemoval.mmar.mmar <image-file> <optional-arguments>
```
where **_<image-file\>_** is the image to process of format _.nii_, _.nii.gz_, or _.tif_.
The optional arguments allow specifying the bval and bvec metadata files as well as the classification method and any associated parameters.  For a full list of command arguments use the "-h" option.


The output is organized by slice, with each output file prepended with the slice name, "Z000", "Z001", "Z002", etc.  Optionally, a directory can be created for each slice, with the prefixes on the output files dropped. The following is the list of files created:
```
<output_dir>
    log.txt
    excluded_frames.csv
    <slice directory>
        <image file> - multi-frame slice image with rejected frames removed
        <bval file> - copy of original bvals file with rejected frames removed
        <noddi bval file> - same as <bval file> with values rounded to nearest 100
        <bvec file> - copy of original bvecs file with rejected frames removed
```
## Sample Test Data
Sample data for three scans has been provided in the *"sample_data"* directory.  When *mmar_gui is started from the dti directory the default config file, *"mmar_project.txt"* will see these scans.  Select a sample scan from the left-hand control, and navigate through the slices and frames using the controls below the image.


## Training Application (motion_detector)
The training program uses a list of classified frames to generate a classification model that is used the the GUI and CLI applications.  The program is invoked to show all options with the following command:
```
python -m MotionArtifactRemoval.motion_detector.run -h
```
The following is a typical command line example:
```
python -m MotionArtifactRemoval.motion_detector.run \
    --mouse_list mice_files.txt \
    --csv_list csv_files.txt
    --output_model_path ./my_model.pkl \
    --train_balanced \
    --start_idx 8 \
    --shift_csv_labels 9 \
    --classifier_names \
        'Logistic Regression' \
        'SVM' 'Random Forest' \
        'AdaBoosted decision tree' \
        'Gradient Boosted Decision Tree' \
        'Gaussian Naive Bayes' \
    --plot_results \
    --search_param
```
The *mouse_list* file is a text file containing a list of scan files that will be used for training.  For example,
```
        data/mouse_085/dwi/data.nii.gz
        data/mouse_086/dwi/data.nii.gz
        data/mouse_087/dwi/data.nii.gz
        data/mouse_088/dwi/data.nii.gz
        data/mouse_089/dwi/data.nii.gz
        data/mouse_090/dwi/data.nii.gz
        data/mouse_091/dwi/data.nii.gz
        data/mouse_092/dwi/data.nii.gz
        data/mouse_094/dwi/data.nii.gz
        data/mouse_095/dwi/data.nii.gz
        data/mouse_096/dwi/data.nii.gz
        data/mouse_097/dwi/data.nii.gz
```
The *csv_list* is similarly a list of corresponding frame classifications, one for each scan. Each file contains line per slice.
Each line contains a list indices of rejected frames.  The following is an example for a scan with 17 slices:
```
21,25,29,34,36,41,46,47,48,51,52,53
34,35,36,41,42,46,47
21,29,30,40,46,47,51,53
35,40,47,52,53
29,34,41,47,51,53
25,29,34,36,40,46,47,51,52,53
26,28,29,34,36,41,46,47,48,51,52,53
21,24,29,31,35,36,41,42,45,47,52,53
14,21,29,34,40,41,46,47,53
34,35,45,46,51,53
34,36,46,51,53
40
31,47,53
29,34,35,42,51,52,53
25,34,35,41,45,46,53
30,41,53
28,46,51,52
```
The training in the example above results in the creation of the model file *my_model.pkl*.  This file can then be selected for
use in the GUI and CLI applications.  The default model file is in the
*MotionArtifactRemoval/motion_detecor/trained_models* directory.
## Training motion detection model
##### 1 - Data Cleaning
For the first batch of data good/bad frame labels were stored in an excel file. These excel files are stored in top level of each mouse directory named as: `2021-03-02_0093_Excl_Dirs_No-Need.xlsx`
We read these excel files and store the labels in a dictionary using `get_rejected_frames_mouse` function and return list of slices containing list of bad frames. In the GUI this extracted list is shown with `"Expert"` button. We use this list to train the model.

##### 2 - Training the model
We train the model using the `train` function under `MotionArtifactRemoval.mmar.motion_detecor`. We use `MotionFrameDetector` class to train the model. This class is able to train `Logistic Regression`, `SVM`, `Random Forest`, `AdaBoosted decision tree`, `Gradient Boosted Decision Tree`, `Gaussian Naive Bayes` models. We recommend `Logistic Regression` model for training as with our current search found to perform robustly across different datasets. `MotionFrameDetector` can train, balance the dataset, plot, and report the result statistics. Options to select different models are only accessible for advanced users through CLI and we only recommend using `Logistic Regression` model for training which is activated for inference and prediction within GUI.

To make the training process faster we use `GridSearchCV` to find the best parameters for the model. A predifined set of parameters are introduced in `MotionArtifactRemoval.mmar.motion_detecor.Parameters` class which could be changed to find the best parameters for the model.

An example of how to trian a new model is provided in `run.py` file. We save the output of the training process in a `.pkl` file. This file is saved in the `MotionArtifactRemoval/motion_detecor/trained_models` directory.

##### 3 - Inference and prediction
CLI and GUI application are provided to perform inference and prediction. The inference and prediction is performed using the trained model. Trained models are saved in the `MotionArtifactRemoval/motion_detecor/trained_models` directory. The core prediction module is currently within `find_bad_frames_ml` function for CLI and `get_rejected_frames_ml_gui` within `MotionArtifactRemoval.mmar.motion_detecor.predict` for GUI. These two will be integrated and refactored to be one. 

## Graphical User Interface Application (mmar_gui)
Run GUI by openning a terminal and running:

```
python -m MotionArtifactRemoval.mmar_gui.gui
```

#### Workflow
##### 1 - Motion artifact detection
Using the CLI application, we can detect the motion artifacts in the DTI and NODDI images. The output is a list of bad frames which is then verfied by the user using the GUI.

##### 2 - Motion artifact removal and scan reconstruction
After the user verifies the bad frames, we remove the bad frames from the DTI and NODDI images and reconstruct the DTI and NODDI images. The reconstructed images are stored in separate `nii` files for each slice where in each slice corresponding bad frames are excluded from all other slices of the 4D image volume.

##### 3 - Downstream Analysis
After creating the `nii` files for each slice, image volumes are processed using the `nipype`(?) package. The `nipype` package is used to perform diffusion analysis.

## random_frames
The script random_frames.py (in MotionArtifactRemoval/mmar/) generates lists of random frames for each slice.  The output can be used as input to mmar.py to generate a scan with random frames removed.  This is used only for testing purposes. The following are example command lines to invoke the program:
```
cd MotionArtifactRemoval/mmar
./random_frames.py -h
./random_frames.py -s 1 -e 46 -n 17 -f 8 -o excluded_frames.csv
```

