#
# - activate environment
# - install latest
# - sample tests
#
conda activate MMAR
pip install ~/work/ImageTools/dti_motion/
cd /Users/peterj/work/CBA/MRI_motion_detection/data/test
python -m MotionArtifactRemoval.mmar.mmar -o CBA_results ../before_Registration/CK_DIF_0001_MUS00028472_1/dti/small_large_2-shells_0.nii.gz
python -m MotionArtifactRemoval.mmar.mmar -o UFL_results ../UFL/20181029_104247_JAX_AD_001_1_1_nifti/dMRI/dMRI.nii.gz
