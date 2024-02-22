python -m MotionArtifactRemoval.motion_detector.run \
    --mouse_list \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX085/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX086/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX087/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX088/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX089/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX090/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX091/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX092/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX094/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX095/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX096/dwi/data.nii.gz \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX097/dwi/data.nii.gz \
    --csv_list \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX085/JAX085_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX086/JAX086_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX087/JAX087_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX088/JAX088_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX089/JAX089_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX090/JAX090_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX091/JAX091_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX092/JAX092_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX094/JAX094_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX095/JAX095_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX096/JAX096_Febo.csv \
        /Users/peterj/work/CBA/MRI_motion_detection/data/UFL/Jax_DWI_annotations/JAX097/JAX097_Febo.csv \
    --output_model_path ./my_model.pkl \
    --start_idx 8 \
    --shift_csv_labels 9 \
    --classifier_names 'Logistic Regression' 'SVM' 'Random Forest' 'AdaBoosted decision tree' 'Gradient Boosted Decision Tree' 'Gaussian Naive Bayes' \
    --plot_results
    #--show_examples
    #--verbose \
    #--search_param \
    #--classifier_names 'Logistic Regression' \

